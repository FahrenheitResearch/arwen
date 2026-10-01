// WRF v4.7.1 module_sf_bep.F, default REAL float32.
// Generated from the pinned Fortran by tools/transcribe_bep.py.
// All indices inside device routines remain Fortran 1-based.
// Arithmetic boundaries are explicit; do not compile with fast math.
// __builtin_offsetof exists in NVRTC 13 and not in NVRTC 12, the CUDA 12
// wheel the default `gpuwm[gpu]` install carries, where every per-member
// assert below failed to compile and options 2/3 could not start.  The
// member offsets are asserted where the builtin exists; the two sizeof
// asserts hold on both toolchains.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
#define BEP_OFFSET_ASSERT(type, member, offset) static_assert(__builtin_offsetof(type, member) == (offset), #member " offset")
#else
#define BEP_OFFSET_ASSERT(type, member, offset) static_assert(true, #member " offset")
#endif
__device__ constexpr int ndm = 2;
__device__ constexpr int nz_um = 18;
__device__ constexpr int ng_u = 10;
__device__ constexpr int nwr_u = 10;
__device__ constexpr int nurbm = 11;
__device__ constexpr int nurbmax = 11;
__device__ constexpr float dz_u = 5.f;
__device__ constexpr float vk = 0.40f;
__device__ constexpr float g_u = 9.81f;
__device__ constexpr float pi = 3.141592653f;
__device__ constexpr float r = 287.f;
__device__ constexpr float cp_u = 1004.f;
__device__ constexpr float rcp_u = 287.f/1004.f;
__device__ constexpr float sigma = 5.67e-08f;
__device__ constexpr float p0 = 1.e+5f;
__device__ constexpr float cdrag = 0.4f;

// module_sf_bep.F:318-346, 589-905; global scratch uses YSU lane placement.
// A block owns [element][32 lanes]. Nested routine frames restore the arena
// cursor on return. No large automatic CUDA arrays or resident-thread-priced
// local backing store. Integer scratch uses the same four-byte slots.
// module_sf_bep.F:3356-3365 zeroes every class of each view factor on each
// column, then only iurb's class slice is used. The views keep the source
// routine's iurb index but alias that last dimension to one private slice.
// Exact class-default view factors are retained in a per-table device cache.
struct BepWS {
    float *p; int pos;
    template<class T> __device__ T *alloc(int n) {
        T *a = reinterpret_cast<T *>(p + (size_t)pos*32); pos += n; return a;
    }
};
struct BepFrame {
    BepWS &ws; int saved;
    __device__ BepFrame(BepWS &w):ws(w),saved(w.pos){}
    __device__ ~BepFrame(){ws.pos=saved;}
};
template<class T,int R> struct BepArray {
    T *p; int lane_stride; int d[R], lo[R]; bool alias_last;
    template<class... A> __device__ BepArray(T *q,int stride,A... args):
        p(q),lane_stride(stride),alias_last(false) {
        int v[] = {int(args)...};
        for(int j=0;j<R;++j){d[j]=v[j];lo[j]=v[R+j];}
        if(sizeof...(args)>2*R) alias_last=v[2*R];
    }
    __device__ int size()const {int n=1;for(int j=0;j<R;++j)n*=d[j];return n;}
    __device__ T &operator[](int i)const{return p[(size_t)i*lane_stride];}
    template<class... A> __device__ T &operator()(A... args)const{
        int v[]={int(args)...}; int index=0, stride=1;
        for(int j=0;j<R;++j){if(!(alias_last && j==R-1))index+=(v[j]-lo[j])*stride;stride*=d[j];}
        return (*this)[index];
    }
};
// module_sf_bep.F:3099, Fortran NINT ties away from zero.
__device__ int bep_nint(float x){return int(x>=0.f?floorf(FADD(x,0.5f)):ceilf(FSUB(x,0.5f)));}
// Table arrays use Fortran storage order, zero-based payload indices:
// X(i,class) -> X[(class-1)*leading_extent + (i-1)]. Scalar-class arrays
// use X[class-1]. MAXDIRS=3, MAXHGTS=50 from module_sf_urban.F:70-72.
// Payload contains the module variables AFTER urban_param_init's unit
// conversion, not raw URBPARM.TBL SI heat capacities/conductivities.
struct UrbanBepTable {
    int icate;
    float capb_tbl[11];
    float capr_tbl[11];
    float capg_tbl[11];
    float aksb_tbl[11];
    float aksr_tbl[11];
    float aksg_tbl[11];
    float tblend_tbl[11];
    float trlend_tbl[11];
    float tglend_tbl[11];
    float albb_tbl[11];
    float albr_tbl[11];
    float albg_tbl[11];
    float epsb_tbl[11];
    float epsr_tbl[11];
    float epsg_tbl[11];
    float z0r_tbl[11];
    float z0g_tbl[11];
    int numdir_tbl[11];
    float street_direction_tbl[33];
    float street_width_tbl[33];
    float building_width_tbl[33];
    int numhgt_tbl[11];
    float height_bin_tbl[550];
    float hpercent_bin_tbl[550];
};
BEP_OFFSET_ASSERT(UrbanBepTable, icate, 0);
BEP_OFFSET_ASSERT(UrbanBepTable, capb_tbl, 4);
BEP_OFFSET_ASSERT(UrbanBepTable, capr_tbl, 48);
BEP_OFFSET_ASSERT(UrbanBepTable, capg_tbl, 92);
BEP_OFFSET_ASSERT(UrbanBepTable, aksb_tbl, 136);
BEP_OFFSET_ASSERT(UrbanBepTable, aksr_tbl, 180);
BEP_OFFSET_ASSERT(UrbanBepTable, aksg_tbl, 224);
BEP_OFFSET_ASSERT(UrbanBepTable, tblend_tbl, 268);
BEP_OFFSET_ASSERT(UrbanBepTable, trlend_tbl, 312);
BEP_OFFSET_ASSERT(UrbanBepTable, tglend_tbl, 356);
BEP_OFFSET_ASSERT(UrbanBepTable, albb_tbl, 400);
BEP_OFFSET_ASSERT(UrbanBepTable, albr_tbl, 444);
BEP_OFFSET_ASSERT(UrbanBepTable, albg_tbl, 488);
BEP_OFFSET_ASSERT(UrbanBepTable, epsb_tbl, 532);
BEP_OFFSET_ASSERT(UrbanBepTable, epsr_tbl, 576);
BEP_OFFSET_ASSERT(UrbanBepTable, epsg_tbl, 620);
BEP_OFFSET_ASSERT(UrbanBepTable, z0r_tbl, 664);
BEP_OFFSET_ASSERT(UrbanBepTable, z0g_tbl, 708);
BEP_OFFSET_ASSERT(UrbanBepTable, numdir_tbl, 752);
BEP_OFFSET_ASSERT(UrbanBepTable, street_direction_tbl, 796);
BEP_OFFSET_ASSERT(UrbanBepTable, street_width_tbl, 928);
BEP_OFFSET_ASSERT(UrbanBepTable, building_width_tbl, 1060);
BEP_OFFSET_ASSERT(UrbanBepTable, numhgt_tbl, 1192);
BEP_OFFSET_ASSERT(UrbanBepTable, height_bin_tbl, 1236);
BEP_OFFSET_ASSERT(UrbanBepTable, hpercent_bin_tbl, 3436);
static_assert(sizeof(UrbanBepTable) == 5636, "UrbanBepTable size");
struct UrbanBepClass {
    float alag_u[11];
    float alaw_u[11];
    float alar_u[11];
    float csg_u[11];
    float csw_u[11];
    float csr_u[11];
    float twini_u[11];
    float trini_u[11];
    float tgini_u[11];
    float albg_u[11];
    float albw_u[11];
    float albr_u[11];
    float emg_u[11];
    float emw_u[11];
    float emr_u[11];
    float z0g_u[11];
    float z0r_u[11];
    int nd_u[11];
    float strd_u[22];
    float drst_u[22];
    float ws_u[22];
    float bs_u[22];
    float h_b[198];
    float d_b[198];
    float ss_u[198];
    float pb_u[198];
    int nz_u[11];
    float z_u[18];
    int error;
};
BEP_OFFSET_ASSERT(UrbanBepClass, alag_u, 0);
BEP_OFFSET_ASSERT(UrbanBepClass, alaw_u, 44);
BEP_OFFSET_ASSERT(UrbanBepClass, alar_u, 88);
BEP_OFFSET_ASSERT(UrbanBepClass, csg_u, 132);
BEP_OFFSET_ASSERT(UrbanBepClass, csw_u, 176);
BEP_OFFSET_ASSERT(UrbanBepClass, csr_u, 220);
BEP_OFFSET_ASSERT(UrbanBepClass, twini_u, 264);
BEP_OFFSET_ASSERT(UrbanBepClass, trini_u, 308);
BEP_OFFSET_ASSERT(UrbanBepClass, tgini_u, 352);
BEP_OFFSET_ASSERT(UrbanBepClass, albg_u, 396);
BEP_OFFSET_ASSERT(UrbanBepClass, albw_u, 440);
BEP_OFFSET_ASSERT(UrbanBepClass, albr_u, 484);
BEP_OFFSET_ASSERT(UrbanBepClass, emg_u, 528);
BEP_OFFSET_ASSERT(UrbanBepClass, emw_u, 572);
BEP_OFFSET_ASSERT(UrbanBepClass, emr_u, 616);
BEP_OFFSET_ASSERT(UrbanBepClass, z0g_u, 660);
BEP_OFFSET_ASSERT(UrbanBepClass, z0r_u, 704);
BEP_OFFSET_ASSERT(UrbanBepClass, nd_u, 748);
BEP_OFFSET_ASSERT(UrbanBepClass, strd_u, 792);
BEP_OFFSET_ASSERT(UrbanBepClass, drst_u, 880);
BEP_OFFSET_ASSERT(UrbanBepClass, ws_u, 968);
BEP_OFFSET_ASSERT(UrbanBepClass, bs_u, 1056);
BEP_OFFSET_ASSERT(UrbanBepClass, h_b, 1144);
BEP_OFFSET_ASSERT(UrbanBepClass, d_b, 1936);
BEP_OFFSET_ASSERT(UrbanBepClass, ss_u, 2728);
BEP_OFFSET_ASSERT(UrbanBepClass, pb_u, 3520);
BEP_OFFSET_ASSERT(UrbanBepClass, nz_u, 4312);
BEP_OFFSET_ASSERT(UrbanBepClass, z_u, 4356);
BEP_OFFSET_ASSERT(UrbanBepClass, error, 4428);
static_assert(sizeof(UrbanBepClass) == 4432, "UrbanBepClass size");

__device__ void bep_bep1d(BepWS &arena, int &error, int iurb, int kms, int kme, int kts, int kte, BepArray<float,1> z, float dt, BepArray<float,1> ua, BepArray<float,1> va, BepArray<float,1> pt, BepArray<float,1> da, BepArray<float,1> pr, BepArray<float,1> pt0, float zr, float deltar, float ah, float rs, float rld, BepArray<float,1> alag, BepArray<float,1> alaw, BepArray<float,1> alar, BepArray<float,1> csg, BepArray<float,1> csw, BepArray<float,1> csr, float albg, float albw, float albr, float emg, float emw, float emr, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,3> fws, BepArray<float,2> fsg, BepArray<float,2> z0, int ndu, BepArray<float,1> strd, BepArray<float,1> drst, BepArray<float,1> ws, BepArray<float,1> bs, BepArray<float,1> ss, BepArray<float,1> pb, int nzu, BepArray<float,1> z_u, BepArray<float,3> tw, BepArray<float,2> tg, BepArray<float,3> tr, BepArray<float,2> sfw, BepArray<float,1> sfg, BepArray<float,2> sfr, BepArray<float,1> a_u, BepArray<float,1> a_v, BepArray<float,1> a_t, BepArray<float,1> a_e, BepArray<float,1> b_u, BepArray<float,1> b_v, BepArray<float,1> b_t, BepArray<float,1> b_e, BepArray<float,1> dlg, BepArray<float,1> dl_u, BepArray<float,1> sf, BepArray<float,1> vl, float &rl_up, float &rs_abs, float &emiss, float &grdflx_urb);
__device__ void bep_param(BepWS &arena, int &error, int iurb, int nzu, int nzurb, int &nzurban, int ndu, BepArray<float,1> csg_u, BepArray<float,1> csg, BepArray<float,1> alag_u, BepArray<float,1> alag, BepArray<float,1> csr_u, BepArray<float,1> csr, BepArray<float,1> alar_u, BepArray<float,1> alar, BepArray<float,1> csw_u, BepArray<float,1> csw, BepArray<float,1> alaw_u, BepArray<float,1> alaw, BepArray<float,2> ws_u, BepArray<float,1> ws, BepArray<float,2> bs_u, BepArray<float,1> bs, BepArray<float,1> z0g_u, BepArray<float,1> z0r_u, BepArray<float,2> z0, BepArray<float,2> strd_u, BepArray<float,1> strd, BepArray<float,2> drst_u, BepArray<float,1> drst, BepArray<float,2> ss_u, BepArray<float,1> ss_urb, BepArray<float,1> ss, BepArray<float,2> pb_u, BepArray<float,1> pb_urb, BepArray<float,1> pb, float lp_urb, float lb_urb, float hgt_urb, float frc_urb);
__device__ void bep_interpol(BepWS &arena, int &error, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z, BepArray<float,1> z_u, BepArray<float,1> c, BepArray<float,1> c_u);
__device__ void bep_modif_rad(BepWS &arena, int &error, int iurb, int nd, int nz_u, BepArray<float,1> z, BepArray<float,1> ws, BepArray<float,1> drst, BepArray<float,1> strd, BepArray<float,1> ss, BepArray<float,1> pb, BepArray<float,3> tw, BepArray<float,2> tg, float albg, float albw, float emw, float emg, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,2> fsg, float zr, float deltar, float ah, float rs, float rl, BepArray<float,2> rsw, BepArray<float,1> rsg, BepArray<float,2> rlw, BepArray<float,1> rlg);
__device__ void bep_surf_temp(BepWS &arena, int &error, int nz_u, int nd, BepArray<float,1> pr, float dt, BepArray<float,1> ss, float rs, float rl, BepArray<float,1> rsg, BepArray<float,1> rlg, BepArray<float,2> rsw, BepArray<float,2> rlw, BepArray<float,2> tg, BepArray<float,1> alag, BepArray<float,1> csg, float emg, float albg, BepArray<float,1> ptg, BepArray<float,1> sfg, BepArray<float,1> gfg, BepArray<float,3> tr, BepArray<float,1> alar, BepArray<float,1> csr, float emr, float albr, BepArray<float,2> ptr, BepArray<float,2> sfr, BepArray<float,2> gfr, BepArray<float,3> tw, BepArray<float,1> alaw, BepArray<float,1> csw, float emw, float albw, BepArray<float,2> ptw, BepArray<float,2> sfw, BepArray<float,2> gfw);
__device__ void bep_buildings(BepWS &arena, int &error, int nd, int nz, BepArray<float,2> z0, BepArray<float,1> ua_u, BepArray<float,1> va_u, BepArray<float,1> pt_u, BepArray<float,1> pt0_u, BepArray<float,1> ptg, BepArray<float,2> ptr, BepArray<float,1> da_u, BepArray<float,2> ptw, BepArray<float,1> drst, BepArray<float,2> uva_u, BepArray<float,2> vva_u, BepArray<float,2> uvb_u, BepArray<float,2> vvb_u, BepArray<float,2> tva_u, BepArray<float,2> tvb_u, BepArray<float,2> evb_u, BepArray<float,2> uhb_u, BepArray<float,2> vhb_u, BepArray<float,2> thb_u, BepArray<float,2> ehb_u, BepArray<float,1> ss, float dt);
__device__ void bep_urban_meso(BepWS &arena, int &error, int nd, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z, BepArray<float,1> dz, BepArray<float,1> z_u, BepArray<float,1> pb, BepArray<float,1> ss, BepArray<float,1> bs, BepArray<float,1> ws, BepArray<float,1> sf, BepArray<float,1> vl, BepArray<float,2> uva_u, BepArray<float,2> vva_u, BepArray<float,2> uvb_u, BepArray<float,2> vvb_u, BepArray<float,2> tva_u, BepArray<float,2> tvb_u, BepArray<float,2> evb_u, BepArray<float,2> uhb_u, BepArray<float,2> vhb_u, BepArray<float,2> thb_u, BepArray<float,2> ehb_u, BepArray<float,1> a_u, BepArray<float,1> a_v, BepArray<float,1> a_t, BepArray<float,1> a_e, BepArray<float,1> b_u, BepArray<float,1> b_v, BepArray<float,1> b_t, BepArray<float,1> b_e);
__device__ void bep_interp_length(BepWS &arena, int &error, int nd, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z_u, BepArray<float,1> z, BepArray<float,1> ss, BepArray<float,1> ws, BepArray<float,1> bs, BepArray<float,1> dlg, BepArray<float,1> dl_u);
__device__ void bep_shadow_mas(BepWS &arena, int &error, int nd, int nz_u, float zr, float deltar, float ah, BepArray<float,1> drst, BepArray<float,1> ws, BepArray<float,1> ss, BepArray<float,1> pb, BepArray<float,1> z, float rs, BepArray<float,2> rsw, BepArray<float,1> rsg);
__device__ void bep_shade_wall(BepWS &arena, int &error, float z1, float z2, float hu, float phix, float aa, float ws, float &rd);
__device__ void bep_long_rad(BepWS &arena, int &error, int iurb, int nz_u, int id, float emw, float emg, BepArray<float,3> fwg, BepArray<float,4> fww, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,2> fsg, BepArray<float,2> tg, BepArray<float,3> tw, BepArray<float,1> rlg, BepArray<float,2> rlw, float rl, BepArray<float,1> pb);
__device__ void bep_short_rad(BepWS &arena, int &error, int iurb, int nz_u, int id, float albw, float albg, BepArray<float,3> fwg, BepArray<float,4> fww, BepArray<float,3> fgw, BepArray<float,1> rsg, BepArray<float,2> rsw, BepArray<float,1> pb);
__device__ void bep_gaussj(BepWS &arena, int &error, BepArray<float,2> a, int n, BepArray<float,1> b, int np);
__device__ void bep_soil_temp(BepWS &arena, int &error, int nz, BepArray<float,1> dz, BepArray<float,1> temp, float &pt, BepArray<float,1> ala, BepArray<float,1> cs, float rs, float rl, float press, float dt, float em, float alb, float &rt, float sf, float &gf);
__device__ void bep_invert(BepWS &arena, int &error, int n, BepArray<float,2> a, BepArray<float,1> c, BepArray<float,1> x);
__device__ void bep_flux_wall(BepWS &arena, int &error, float ua, float va, float pt, float da, float ptw, float &uva, float &vva, float &uvb, float &vvb, float &tva, float &tvb, float &evb, float drst, float dt);
__device__ void bep_flux_flat(BepWS &arena, int &error, float dz, float z0, float ua, float va, float pt, float pt0, float ptg, float &uhb, float &vhb, float &thb, float &ehb);
__device__ void bep_icbep(BepWS &arena, int &error, BepArray<int,1> nd_u, BepArray<float,2> h_b, BepArray<float,2> d_b, BepArray<float,2> ss_u, BepArray<float,2> pb_u, BepArray<int,1> nz_u, BepArray<float,1> z_u);
__device__ void bep_view_factors(BepWS &arena, int &error, int iurb, int nz_u, int id, float dxy, BepArray<float,1> z, float ws, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,2> fsg, BepArray<float,3> fsw, BepArray<float,3> fws);
__device__ void bep_fprls(BepWS &arena, int &error, float &fprl, float a, float b, float c);
__device__ void bep_fnrms(BepWS &arena, int &error, float &fnrm, float a, float b, float c);
__device__ void bep_init_para(BepWS &arena, int &error, const UrbanBepTable &tbl, BepArray<float,1> alag_u, BepArray<float,1> alaw_u, BepArray<float,1> alar_u, BepArray<float,1> csg_u, BepArray<float,1> csw_u, BepArray<float,1> csr_u, BepArray<float,1> twini_u, BepArray<float,1> trini_u, BepArray<float,1> tgini_u, BepArray<float,1> albg_u, BepArray<float,1> albw_u, BepArray<float,1> albr_u, BepArray<float,1> emg_u, BepArray<float,1> emw_u, BepArray<float,1> emr_u, BepArray<float,1> z0g_u, BepArray<float,1> z0r_u, BepArray<int,1> nd_u, BepArray<float,2> strd_u, BepArray<float,2> drst_u, BepArray<float,2> ws_u, BepArray<float,2> bs_u, BepArray<float,2> h_b, BepArray<float,2> d_b);
__device__ void bep_upward_rad(BepWS &arena, int &error, int ndu, int nzu, BepArray<float,1> ws, BepArray<float,1> bs, float sigma, BepArray<float,1> pb, BepArray<float,1> ss, BepArray<float,2> tg, float emg_u, float albg_u, BepArray<float,1> rlg, BepArray<float,1> rsg, BepArray<float,1> sfg, BepArray<float,3> tw, float emw_u, float albw_u, BepArray<float,2> rlw, BepArray<float,2> rsw, BepArray<float,2> sfw, BepArray<float,3> tr, float emr_u, float albr_u, float rld, float rs, BepArray<float,2> sfr, float &rs_abs, float &rl_up, float &emiss, float &grdflx_urb);
__device__ void bep_icbep_xy(BepWS &arena, int &error, int iurb, BepArray<float,4> fww_u, BepArray<float,3> fwg_u, BepArray<float,3> fgw_u, BepArray<float,3> fsw_u, BepArray<float,3> fws_u, BepArray<float,2> fsg_u, int ndu, BepArray<float,1> strd, BepArray<float,1> ws, int nzu, BepArray<float,1> z_u);
__device__ void bep_icbephi_xy(BepWS &arena, int &error, BepArray<float,1> hb_u, BepArray<float,1> hi_urb1d, BepArray<float,1> ss_u, BepArray<float,1> pb_u, int &nzu, BepArray<float,1> z_u);

// module_sf_bep.F:589-905
__device__ void bep_bep1d(BepWS &arena, int &error, int iurb, int kms, int kme, int kts, int kte, BepArray<float,1> z, float dt, BepArray<float,1> ua, BepArray<float,1> va, BepArray<float,1> pt, BepArray<float,1> da, BepArray<float,1> pr, BepArray<float,1> pt0, float zr, float deltar, float ah, float rs, float rld, BepArray<float,1> alag, BepArray<float,1> alaw, BepArray<float,1> alar, BepArray<float,1> csg, BepArray<float,1> csw, BepArray<float,1> csr, float albg, float albw, float albr, float emg, float emw, float emr, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,3> fws, BepArray<float,2> fsg, BepArray<float,2> z0, int ndu, BepArray<float,1> strd, BepArray<float,1> drst, BepArray<float,1> ws, BepArray<float,1> bs, BepArray<float,1> ss, BepArray<float,1> pb, int nzu, BepArray<float,1> z_u, BepArray<float,3> tw, BepArray<float,2> tg, BepArray<float,3> tr, BepArray<float,2> sfw, BepArray<float,1> sfg, BepArray<float,2> sfr, BepArray<float,1> a_u, BepArray<float,1> a_v, BepArray<float,1> a_t, BepArray<float,1> a_e, BepArray<float,1> b_u, BepArray<float,1> b_v, BepArray<float,1> b_t, BepArray<float,1> b_e, BepArray<float,1> dlg, BepArray<float,1> dl_u, BepArray<float,1> sf, BepArray<float,1> vl, float &rl_up, float &rs_abs, float &emiss, float &grdflx_urb) {
    BepFrame frame(arena);
    BepArray<float,1> gfg(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> gfr(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> gfw(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,1> dz(arena.alloc<float>(((kme-kms+1))), 32, (kme-kms+1), kms);
    BepArray<float,1> ua_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> va_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> pt_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> da_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> pt0_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> pr_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> rsg(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> rsw(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,1> rlg(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> rlw(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,1> ptg(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> ptr(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> ptw(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> uhb_u(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> uva_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> uvb_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> vhb_u(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> vva_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> vvb_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> thb_u(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> tva_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> tvb_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,2> ehb_u(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> evb_u(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    int iz;
    int id;
    int iw;
    int ix;
    int iy;
    // module_sf_bep.F:811
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:812
    dz(iz) = FSUB(z((iz + 1)), z(iz));
    // module_sf_bep.F:813
    }
    // module_sf_bep.F:816
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, ua, ua_u); if (error) return;
    // module_sf_bep.F:817
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, va, va_u); if (error) return;
    // module_sf_bep.F:818
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, pt, pt_u); if (error) return;
    // module_sf_bep.F:819
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, pt0, pt0_u); if (error) return;
    // module_sf_bep.F:820
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, pr, pr_u); if (error) return;
    // module_sf_bep.F:821
    bep_interpol(arena, error, kms, kme, kts, kte, nzu, z, z_u, da, da_u); if (error) return;
    // module_sf_bep.F:826
    bep_modif_rad(arena, error, iurb, ndu, nzu, z_u, ws, drst, strd, ss, pb, tw, tg, albg, albw, emw, emg, fww, fwg, fgw, fsw, fsg, zr, deltar, ah, rs, rld, rsw, rsg, rlw, rlg); if (error) return;
    // module_sf_bep.F:835
    bep_upward_rad(arena, error, ndu, nzu, ws, bs, sigma, pb, ss, tg, emg, albg, rlg, rsg, sfg, tw, emw, albw, rlw, rsw, sfw, tr, emr, albr, rld, rs, sfr, rs_abs, rl_up, emiss, grdflx_urb); if (error) return;
    // module_sf_bep.F:845
    bep_surf_temp(arena, error, nzu, ndu, pr_u, dt, ss, rs, rld, rsg, rlg, rsw, rlw, tg, alag, csg, emg, albg, ptg, sfg, gfg, tr, alar, csr, emr, albr, ptr, sfr, gfr, tw, alaw, csw, emw, albw, ptw, sfw, gfw); if (error) return;
    // module_sf_bep.F:854
    bep_buildings(arena, error, ndu, nzu, z0, ua_u, va_u, pt_u, pt0_u, ptg, ptr, da_u, ptw, drst, uva_u, vva_u, uvb_u, vvb_u, tva_u, tvb_u, evb_u, uhb_u, vhb_u, thb_u, ehb_u, ss, dt); if (error) return;
    // module_sf_bep.F:866
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:867
    sfg(id) = FMUL(FMUL((-da_u(1)), cp_u), thb_u(id, 1));
    // module_sf_bep.F:868
    for (iz = 2; iz <= nzu; iz += 1) {
    // module_sf_bep.F:869
    sfr(id, iz) = FMUL(FMUL((-da_u(iz)), cp_u), thb_u(id, iz));
    // module_sf_bep.F:870
    }
    // module_sf_bep.F:872
    for (iz = 1; iz <= nzu; iz += 1) {
    // module_sf_bep.F:873
    sfw(((2 * id) - 1), iz) = FMUL(FMUL((-da_u(iz)), cp_u), (FADD(tvb_u(((2 * id) - 1), iz), FMUL(tva_u(((2 * id) - 1), iz), pt_u(iz)))));
    // module_sf_bep.F:875
    sfw((2 * id), iz) = FMUL(FMUL((-da_u(iz)), cp_u), (FADD(tvb_u((2 * id), iz), FMUL(tva_u((2 * id), iz), pt_u(iz)))));
    // module_sf_bep.F:877
    }
    // module_sf_bep.F:878
    }
    // module_sf_bep.F:891
    bep_urban_meso(arena, error, ndu, kms, kme, kts, kte, nzu, z, dz, z_u, pb, ss, bs, ws, sf, vl, uva_u, vva_u, uvb_u, vvb_u, tva_u, tvb_u, evb_u, uhb_u, vhb_u, thb_u, ehb_u, a_u, a_v, a_t, a_e, b_u, b_v, b_t, b_e); if (error) return;
    // module_sf_bep.F:902
    bep_interp_length(arena, error, ndu, kms, kme, kts, kte, nzu, z_u, z, ss, ws, bs, dlg, dl_u); if (error) return;
    // module_sf_bep.F:904
    return;
}

// module_sf_bep.F:910-1075
__device__ void bep_param(BepWS &arena, int &error, int iurb, int nzu, int nzurb, int &nzurban, int ndu, BepArray<float,1> csg_u, BepArray<float,1> csg, BepArray<float,1> alag_u, BepArray<float,1> alag, BepArray<float,1> csr_u, BepArray<float,1> csr, BepArray<float,1> alar_u, BepArray<float,1> alar, BepArray<float,1> csw_u, BepArray<float,1> csw, BepArray<float,1> alaw_u, BepArray<float,1> alaw, BepArray<float,2> ws_u, BepArray<float,1> ws, BepArray<float,2> bs_u, BepArray<float,1> bs, BepArray<float,1> z0g_u, BepArray<float,1> z0r_u, BepArray<float,2> z0, BepArray<float,2> strd_u, BepArray<float,1> strd, BepArray<float,2> drst_u, BepArray<float,1> drst, BepArray<float,2> ss_u, BepArray<float,1> ss_urb, BepArray<float,1> ss, BepArray<float,2> pb_u, BepArray<float,1> pb_urb, BepArray<float,1> pb, float lp_urb, float lb_urb, float hgt_urb, float frc_urb) {
    BepFrame frame(arena);
    int id;
    int ig;
    int ir;
    int iw;
    int iz;
    int ihu;
    // module_sf_bep.F:981
    for (int wi=0; wi<ss.size(); ++wi) ss[wi] = 0.f;
    // module_sf_bep.F:982
    for (int wi=0; wi<pb.size(); ++wi) pb[wi] = 0.f;
    // module_sf_bep.F:983
    for (int wi=0; wi<csg.size(); ++wi) csg[wi] = 0.f;
    // module_sf_bep.F:984
    for (int wi=0; wi<alag.size(); ++wi) alag[wi] = 0.f;
    // module_sf_bep.F:985
    for (int wi=0; wi<csr.size(); ++wi) csr[wi] = 0.f;
    // module_sf_bep.F:986
    for (int wi=0; wi<alar.size(); ++wi) alar[wi] = 0.f;
    // module_sf_bep.F:987
    for (int wi=0; wi<csw.size(); ++wi) csw[wi] = 0.f;
    // module_sf_bep.F:988
    for (int wi=0; wi<alaw.size(); ++wi) alaw[wi] = 0.f;
    // module_sf_bep.F:989
    for (int wi=0; wi<z0.size(); ++wi) z0[wi] = 0.f;
    // module_sf_bep.F:990
    for (int wi=0; wi<ws.size(); ++wi) ws[wi] = 0.f;
    // module_sf_bep.F:991
    for (int wi=0; wi<bs.size(); ++wi) bs[wi] = 0.f;
    // module_sf_bep.F:992
    for (int wi=0; wi<strd.size(); ++wi) strd[wi] = 0.f;
    // module_sf_bep.F:993
    for (int wi=0; wi<drst.size(); ++wi) drst[wi] = 0.f;
    // module_sf_bep.F:994
    nzurban = 0;
    // module_sf_bep.F:996
    ihu = 0;
    // module_sf_bep.F:998
    for (iz = 1; iz <= nz_um; iz += 1) {
    // module_sf_bep.F:999
    if ((ss_urb(iz) != 0.f)) {
    // module_sf_bep.F:1000
    ihu = 1;
    // module_sf_bep.F:1001
    break;
    // module_sf_bep.F:1002
    } else {
    // module_sf_bep.F:1003
    ;
    // module_sf_bep.F:1004
    }
    // module_sf_bep.F:1005
    }
    // module_sf_bep.F:1007
    if ((ihu == 1)) {
    // module_sf_bep.F:1008
    for (iz = 1; iz <= (nzurb + 1); iz += 1) {
    // module_sf_bep.F:1009
    ss(iz) = ss_urb(iz);
    // module_sf_bep.F:1010
    pb(iz) = pb_urb(iz);
    // module_sf_bep.F:1011
    }
    // module_sf_bep.F:1012
    nzurban = nzurb;
    // module_sf_bep.F:1013
    } else {
    // module_sf_bep.F:1014
    for (iz = 1; iz <= (nzu + 1); iz += 1) {
    // module_sf_bep.F:1015
    ss(iz) = ss_u(iz, iurb);
    // module_sf_bep.F:1016
    pb(iz) = pb_u(iz, iurb);
    // module_sf_bep.F:1017
    }
    // module_sf_bep.F:1018
    nzurban = nzu;
    // module_sf_bep.F:1019
    }
    // module_sf_bep.F:1021
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:1022
    z0(id, 1) = z0g_u(iurb);
    // module_sf_bep.F:1023
    for (iz = 2; iz <= (nzurban + 1); iz += 1) {
    // module_sf_bep.F:1024
    z0(id, iz) = z0r_u(iurb);
    // module_sf_bep.F:1025
    }
    // module_sf_bep.F:1026
    }
    // module_sf_bep.F:1028
    for (ig = 1; ig <= ng_u; ig += 1) {
    // module_sf_bep.F:1029
    csg(ig) = csg_u(iurb);
    // module_sf_bep.F:1030
    alag(ig) = alag_u(iurb);
    // module_sf_bep.F:1031
    }
    // module_sf_bep.F:1033
    for (ir = 1; ir <= nwr_u; ir += 1) {
    // module_sf_bep.F:1034
    csr(ir) = csr_u(iurb);
    // module_sf_bep.F:1035
    alar(ir) = alar_u(iurb);
    // module_sf_bep.F:1036
    }
    // module_sf_bep.F:1038
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:1039
    csw(iw) = csw_u(iurb);
    // module_sf_bep.F:1040
    alaw(iw) = alaw_u(iurb);
    // module_sf_bep.F:1041
    }
    // module_sf_bep.F:1043
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:1044
    strd(id) = strd_u(id, iurb);
    // module_sf_bep.F:1045
    drst(id) = drst_u(id, iurb);
    // module_sf_bep.F:1046
    }
    // module_sf_bep.F:1048
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:1049
    if (((((hgt_urb <= 0.f)) || ((lp_urb <= 0.f))) || ((lb_urb <= 0.f)))) {
    // module_sf_bep.F:1050
    ws(id) = ws_u(id, iurb);
    // module_sf_bep.F:1051
    bs(id) = bs_u(id, iurb);
    // module_sf_bep.F:1052
    } else if ((((FDIV(lp_urb, frc_urb) < 1.f)) && ((lp_urb < lb_urb)))) {
    // module_sf_bep.F:1053
    bs(id) = FDIV(FMUL(FMUL(2.f, hgt_urb), lp_urb), (FSUB(lb_urb, lp_urb)));
    // module_sf_bep.F:1054
    ws(id) = FDIV(FMUL(FMUL(FMUL(2.f, hgt_urb), lp_urb), (FSUB((FDIV(frc_urb, lp_urb)), 1.f))), (FSUB(lb_urb, lp_urb)));
    // module_sf_bep.F:1055
    } else {
    // module_sf_bep.F:1056
    ws(id) = ws_u(id, iurb);
    // module_sf_bep.F:1057
    bs(id) = bs_u(id, iurb);
    // module_sf_bep.F:1058
    }
    // module_sf_bep.F:1059
    }
    // module_sf_bep.F:1060
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:1061
    if ((((bs(id) <= 1.f)) || ((bs(id) >= 150.f)))) {
    // module_sf_bep.F:1064
    bs(id) = bs_u(id, iurb);
    // module_sf_bep.F:1065
    ws(id) = ws_u(id, iurb);
    // module_sf_bep.F:1066
    }
    // module_sf_bep.F:1067
    if ((((ws(id) <= 1.f)) || ((ws(id) >= 150.f)))) {
    // module_sf_bep.F:1070
    bs(id) = bs_u(id, iurb);
    // module_sf_bep.F:1071
    ws(id) = ws_u(id, iurb);
    // module_sf_bep.F:1072
    }
    // module_sf_bep.F:1073
    }
    // module_sf_bep.F:1074
    return;
}

// module_sf_bep.F:1080-1128
__device__ void bep_interpol(BepWS &arena, int &error, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z, BepArray<float,1> z_u, BepArray<float,1> c, BepArray<float,1> c_u) {
    BepFrame frame(arena);
    int iz_u;
    int iz;
    float ctot;
    float dz;
    // module_sf_bep.F:1118
    for (iz_u = 1; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1119
    ctot = 0.f;
    // module_sf_bep.F:1120
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1121
    dz = fmaxf(FSUB(fminf(z((iz + 1)), z_u((iz_u + 1))), fmaxf(z(iz), z_u(iz_u))), 0.f);
    // module_sf_bep.F:1122
    ctot = FADD(ctot, FMUL(c(iz), dz));
    // module_sf_bep.F:1123
    }
    // module_sf_bep.F:1124
    c_u(iz_u) = FDIV(ctot, (FSUB(z_u((iz_u + 1)), z_u(iz_u))));
    // module_sf_bep.F:1125
    }
    // module_sf_bep.F:1127
    return;
}

// module_sf_bep.F:1133-1205
__device__ void bep_modif_rad(BepWS &arena, int &error, int iurb, int nd, int nz_u, BepArray<float,1> z, BepArray<float,1> ws, BepArray<float,1> drst, BepArray<float,1> strd, BepArray<float,1> ss, BepArray<float,1> pb, BepArray<float,3> tw, BepArray<float,2> tg, float albg, float albw, float emw, float emg, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,2> fsg, float zr, float deltar, float ah, float rs, float rl, BepArray<float,2> rsw, BepArray<float,1> rsg, BepArray<float,2> rlw, BepArray<float,1> rlg) {
    BepFrame frame(arena);
    BepArray<float,3> fws(arena.alloc<float>((nz_um) * (ndm) * (1)), 32, nz_um, ndm, 1, 1, 1, 1, true);
    int id;
    int iz;
    // module_sf_bep.F:1192
    bep_shadow_mas(arena, error, nd, nz_u, zr, deltar, ah, drst, ws, ss, pb, z, rs, rsw, rsg); if (error) return;
    // module_sf_bep.F:1196
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1197
    bep_long_rad(arena, error, iurb, nz_u, id, emw, emg, fwg, fww, fgw, fsw, fsg, tg, tw, rlg, rlw, rl, pb); if (error) return;
    // module_sf_bep.F:1200
    bep_short_rad(arena, error, iurb, nz_u, id, albw, albg, fwg, fww, fgw, rsg, rsw, pb); if (error) return;
    // module_sf_bep.F:1202
    }
    // module_sf_bep.F:1204
    return;
}

// module_sf_bep.F:1211-1367
__device__ void bep_surf_temp(BepWS &arena, int &error, int nz_u, int nd, BepArray<float,1> pr, float dt, BepArray<float,1> ss, float rs, float rl, BepArray<float,1> rsg, BepArray<float,1> rlg, BepArray<float,2> rsw, BepArray<float,2> rlw, BepArray<float,2> tg, BepArray<float,1> alag, BepArray<float,1> csg, float emg, float albg, BepArray<float,1> ptg, BepArray<float,1> sfg, BepArray<float,1> gfg, BepArray<float,3> tr, BepArray<float,1> alar, BepArray<float,1> csr, float emr, float albr, BepArray<float,2> ptr, BepArray<float,2> sfr, BepArray<float,2> gfr, BepArray<float,3> tw, BepArray<float,1> alaw, BepArray<float,1> csw, float emw, float albw, BepArray<float,2> ptw, BepArray<float,2> sfw, BepArray<float,2> gfw) {
    BepFrame frame(arena);
    int id;
    int ig;
    int ir;
    int iw;
    int iz;
    BepArray<float,1> rtg(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> rtr(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,2> rtw(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,1> tg_tmp(arena.alloc<float>((ng_u)), 32, ng_u, 1);
    BepArray<float,1> tr_tmp(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> tw_tmp(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> dzg_u(arena.alloc<float>((ng_u)), 32, ng_u, 1);
    BepArray<float,1> dzr_u(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> dzw_u(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    dzg_u[0] = 0.2f;
    dzg_u[1] = 0.12f;
    dzg_u[2] = 0.08f;
    dzg_u[3] = 0.05f;
    dzg_u[4] = 0.03f;
    dzg_u[5] = 0.02f;
    dzg_u[6] = 0.02f;
    dzg_u[7] = 0.01f;
    dzg_u[8] = 0.005f;
    dzg_u[9] = 0.0025f;
    dzr_u[0] = 0.02f;
    dzr_u[1] = 0.02f;
    dzr_u[2] = 0.02f;
    dzr_u[3] = 0.02f;
    dzr_u[4] = 0.02f;
    dzr_u[5] = 0.02f;
    dzr_u[6] = 0.02f;
    dzr_u[7] = 0.01f;
    dzr_u[8] = 0.005f;
    dzr_u[9] = 0.0025f;
    dzw_u[0] = 0.02f;
    dzw_u[1] = 0.02f;
    dzw_u[2] = 0.02f;
    dzw_u[3] = 0.02f;
    dzw_u[4] = 0.02f;
    dzw_u[5] = 0.02f;
    dzw_u[6] = 0.02f;
    dzw_u[7] = 0.01f;
    dzw_u[8] = 0.005f;
    dzw_u[9] = 0.0025f;
    // module_sf_bep.F:1297
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1300
    for (ig = 1; ig <= ng_u; ig += 1) {
    // module_sf_bep.F:1301
    tg_tmp(ig) = tg(id, ig);
    // module_sf_bep.F:1302
    }
    // module_sf_bep.F:1304
    bep_soil_temp(arena, error, ng_u, dzg_u, tg_tmp, ptg(id), alag, csg, rsg(id), rlg(id), pr(1), dt, emg, albg, rtg(id), sfg(id), gfg(id)); if (error) return;
    // module_sf_bep.F:1308
    for (ig = 1; ig <= ng_u; ig += 1) {
    // module_sf_bep.F:1309
    tg(id, ig) = tg_tmp(ig);
    // module_sf_bep.F:1310
    }
    // module_sf_bep.F:1314
    for (iz = 2; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:1316
    if ((ss(iz) > 0.f)) {
    // module_sf_bep.F:1317
    for (ir = 1; ir <= nwr_u; ir += 1) {
    // module_sf_bep.F:1318
    tr_tmp(ir) = tr(id, iz, ir);
    // module_sf_bep.F:1319
    }
    // module_sf_bep.F:1321
    bep_soil_temp(arena, error, nwr_u, dzr_u, tr_tmp, ptr(id, iz), alar, csr, rs, rl, pr(iz), dt, emr, albr, rtr(id, iz), sfr(id, iz), gfr(id, iz)); if (error) return;
    // module_sf_bep.F:1324
    for (ir = 1; ir <= nwr_u; ir += 1) {
    // module_sf_bep.F:1325
    tr(id, iz, ir) = tr_tmp(ir);
    // module_sf_bep.F:1326
    }
    // module_sf_bep.F:1328
    }
    // module_sf_bep.F:1330
    }
    // module_sf_bep.F:1334
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:1336
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:1337
    tw_tmp(iw) = tw(((2 * id) - 1), iz, iw);
    // module_sf_bep.F:1338
    }
    // module_sf_bep.F:1339
    bep_soil_temp(arena, error, nwr_u, dzw_u, tw_tmp, ptw(((2 * id) - 1), iz), alaw, csw, rsw(((2 * id) - 1), iz), rlw(((2 * id) - 1), iz), pr(iz), dt, emw, albw, rtw(((2 * id) - 1), iz), sfw(((2 * id) - 1), iz), gfw(((2 * id) - 1), iz)); if (error) return;
    // module_sf_bep.F:1345
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:1346
    tw(((2 * id) - 1), iz, iw) = tw_tmp(iw);
    // module_sf_bep.F:1347
    }
    // module_sf_bep.F:1349
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:1350
    tw_tmp(iw) = tw((2 * id), iz, iw);
    // module_sf_bep.F:1351
    }
    // module_sf_bep.F:1353
    bep_soil_temp(arena, error, nwr_u, dzw_u, tw_tmp, ptw((2 * id), iz), alaw, csw, rsw((2 * id), iz), rlw((2 * id), iz), pr(iz), dt, emw, albw, rtw((2 * id), iz), sfw((2 * id), iz), gfw((2 * id), iz)); if (error) return;
    // module_sf_bep.F:1358
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:1359
    tw((2 * id), iz, iw) = tw_tmp(iw);
    // module_sf_bep.F:1360
    }
    // module_sf_bep.F:1362
    }
    // module_sf_bep.F:1364
    }
    // module_sf_bep.F:1366
    return;
}

// module_sf_bep.F:1372-1481
__device__ void bep_buildings(BepWS &arena, int &error, int nd, int nz, BepArray<float,2> z0, BepArray<float,1> ua_u, BepArray<float,1> va_u, BepArray<float,1> pt_u, BepArray<float,1> pt0_u, BepArray<float,1> ptg, BepArray<float,2> ptr, BepArray<float,1> da_u, BepArray<float,2> ptw, BepArray<float,1> drst, BepArray<float,2> uva_u, BepArray<float,2> vva_u, BepArray<float,2> uvb_u, BepArray<float,2> vvb_u, BepArray<float,2> tva_u, BepArray<float,2> tvb_u, BepArray<float,2> evb_u, BepArray<float,2> uhb_u, BepArray<float,2> vhb_u, BepArray<float,2> thb_u, BepArray<float,2> ehb_u, BepArray<float,1> ss, float dt) {
    BepFrame frame(arena);
    float dz;
    int id;
    int iz;
    // module_sf_bep.F:1435
    dz = dz_u;
    // module_sf_bep.F:1437
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1440
    bep_flux_flat(arena, error, dz, z0(id, 1), ua_u(1), va_u(1), pt_u(1), pt0_u(1), ptg(id), uhb_u(id, 1), vhb_u(id, 1), thb_u(id, 1), ehb_u(id, 1)); if (error) return;
    // module_sf_bep.F:1445
    for (iz = 2; iz <= nz; iz += 1) {
    // module_sf_bep.F:1446
    if ((ss(iz) > 0)) {
    // module_sf_bep.F:1447
    bep_flux_flat(arena, error, dz, z0(id, iz), ua_u(iz), va_u(iz), pt_u(iz), pt0_u(iz), ptr(id, iz), uhb_u(id, iz), vhb_u(id, iz), thb_u(id, iz), ehb_u(id, iz)); if (error) return;
    // module_sf_bep.F:1451
    } else {
    // module_sf_bep.F:1452
    uhb_u(id, iz) = 0.0f;
    // module_sf_bep.F:1453
    vhb_u(id, iz) = 0.0f;
    // module_sf_bep.F:1454
    thb_u(id, iz) = 0.0f;
    // module_sf_bep.F:1455
    ehb_u(id, iz) = 0.0f;
    // module_sf_bep.F:1456
    }
    // module_sf_bep.F:1457
    }
    // module_sf_bep.F:1460
    for (iz = 1; iz <= nz; iz += 1) {
    // module_sf_bep.F:1461
    bep_flux_wall(arena, error, ua_u(iz), va_u(iz), pt_u(iz), da_u(iz), ptw(((2 * id) - 1), iz), uva_u(((2 * id) - 1), iz), vva_u(((2 * id) - 1), iz), uvb_u(((2 * id) - 1), iz), vvb_u(((2 * id) - 1), iz), tva_u(((2 * id) - 1), iz), tvb_u(((2 * id) - 1), iz), evb_u(((2 * id) - 1), iz), drst(id), dt); if (error) return;
    // module_sf_bep.F:1468
    bep_flux_wall(arena, error, ua_u(iz), va_u(iz), pt_u(iz), da_u(iz), ptw((2 * id), iz), uva_u((2 * id), iz), vva_u((2 * id), iz), uvb_u((2 * id), iz), vvb_u((2 * id), iz), tva_u((2 * id), iz), tvb_u((2 * id), iz), evb_u((2 * id), iz), drst(id), dt); if (error) return;
    // module_sf_bep.F:1476
    }
    // module_sf_bep.F:1478
    }
    // module_sf_bep.F:1480
    return;
}

// module_sf_bep.F:1487-1683
__device__ void bep_urban_meso(BepWS &arena, int &error, int nd, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z, BepArray<float,1> dz, BepArray<float,1> z_u, BepArray<float,1> pb, BepArray<float,1> ss, BepArray<float,1> bs, BepArray<float,1> ws, BepArray<float,1> sf, BepArray<float,1> vl, BepArray<float,2> uva_u, BepArray<float,2> vva_u, BepArray<float,2> uvb_u, BepArray<float,2> vvb_u, BepArray<float,2> tva_u, BepArray<float,2> tvb_u, BepArray<float,2> evb_u, BepArray<float,2> uhb_u, BepArray<float,2> vhb_u, BepArray<float,2> thb_u, BepArray<float,2> ehb_u, BepArray<float,1> a_u, BepArray<float,1> a_v, BepArray<float,1> a_t, BepArray<float,1> a_e, BepArray<float,1> b_u, BepArray<float,1> b_v, BepArray<float,1> b_t, BepArray<float,1> b_e) {
    BepFrame frame(arena);
    float dzz;
    float fact;
    int id;
    int iz;
    int iz_u;
    float se;
    float sr;
    float st;
    float su;
    float sv;
    BepArray<float,1> uet(arena.alloc<float>(((kme-kms+1))), 32, (kme-kms+1), kms);
    float veb;
    float vta;
    float vtb;
    float vte;
    float vtot;
    float vua;
    float vub;
    float vva;
    float vvb;
    // module_sf_bep.F:1560
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1561
    a_u(iz) = 0.f;
    // module_sf_bep.F:1562
    a_v(iz) = 0.f;
    // module_sf_bep.F:1563
    a_t(iz) = 0.f;
    // module_sf_bep.F:1564
    a_e(iz) = 0.f;
    // module_sf_bep.F:1565
    b_u(iz) = 0.f;
    // module_sf_bep.F:1566
    b_v(iz) = 0.f;
    // module_sf_bep.F:1567
    b_e(iz) = 0.f;
    // module_sf_bep.F:1568
    b_t(iz) = 0.f;
    // module_sf_bep.F:1569
    uet(iz) = 0.f;
    // module_sf_bep.F:1570
    }
    // module_sf_bep.F:1573
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1574
    sf(iz) = 0.f;
    // module_sf_bep.F:1575
    vl(iz) = 0.f;
    // module_sf_bep.F:1576
    }
    // module_sf_bep.F:1577
    sf((kte + 1)) = 0.f;
    // module_sf_bep.F:1579
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1580
    for (iz = (kts + 1); iz <= (kte + 1); iz += 1) {
    // module_sf_bep.F:1581
    sr = 0.f;
    // module_sf_bep.F:1582
    for (iz_u = 2; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1583
    if (((z(iz) < z_u(iz_u)) && (z(iz) >= z_u((iz_u - 1))))) {
    // module_sf_bep.F:1584
    sr = pb(iz_u);
    // module_sf_bep.F:1585
    }
    // module_sf_bep.F:1586
    }
    // module_sf_bep.F:1587
    sf(iz) = FADD(sf(iz), FDIV((FDIV((FADD(ws(id), FMUL((FSUB(1.f, sr)), bs(id)))), (FADD(ws(id), bs(id))))), float(nd)));
    // module_sf_bep.F:1588
    }
    // module_sf_bep.F:1589
    }
    // module_sf_bep.F:1592
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1593
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1594
    vtot = 0.f;
    // module_sf_bep.F:1595
    for (iz_u = 1; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1596
    dzz = fmaxf(FSUB(fminf(z_u((iz_u + 1)), z((iz + 1))), fmaxf(z_u(iz_u), z(iz))), 0.f);
    // module_sf_bep.F:1597
    vtot = FADD(vtot, FMUL(pb((iz_u + 1)), dzz));
    // module_sf_bep.F:1598
    }
    // module_sf_bep.F:1599
    vtot = FDIV(vtot, (FSUB(z((iz + 1)), z(iz))));
    // module_sf_bep.F:1600
    vl(iz) = FADD(vl(iz), FDIV((FSUB(1.f, FDIV(FMUL(vtot, bs(id)), (FADD(ws(id), bs(id)))))), float(nd)));
    // module_sf_bep.F:1601
    }
    // module_sf_bep.F:1602
    }
    // module_sf_bep.F:1606
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1608
    fact = FDIV(FDIV(FMUL(FDIV(FDIV(1.f, vl(kts)), dz(kts)), ws(id)), (FADD(ws(id), bs(id)))), float(nd));
    // module_sf_bep.F:1609
    b_t(kts) = FADD(b_t(kts), FMUL(thb_u(id, 1), fact));
    // module_sf_bep.F:1610
    b_u(kts) = FADD(b_u(kts), FMUL(uhb_u(id, 1), fact));
    // module_sf_bep.F:1611
    b_v(kts) = FADD(b_v(kts), FMUL(vhb_u(id, 1), fact));
    // module_sf_bep.F:1612
    b_e(kts) = FADD(b_e(kts), FMUL(FMUL(ehb_u(id, 1), fact), (FSUB(z_u(2), z_u(1)))));
    // module_sf_bep.F:1614
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1615
    st = 0.f;
    // module_sf_bep.F:1616
    su = 0.f;
    // module_sf_bep.F:1617
    sv = 0.f;
    // module_sf_bep.F:1618
    se = 0.f;
    // module_sf_bep.F:1619
    for (iz_u = 2; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1620
    if (((z(iz) <= z_u(iz_u)) && (z((iz + 1)) > z_u(iz_u)))) {
    // module_sf_bep.F:1621
    st = FADD(st, FMUL(ss(iz_u), thb_u(id, iz_u)));
    // module_sf_bep.F:1622
    su = FADD(su, FMUL(ss(iz_u), uhb_u(id, iz_u)));
    // module_sf_bep.F:1623
    sv = FADD(sv, FMUL(ss(iz_u), vhb_u(id, iz_u)));
    // module_sf_bep.F:1624
    se = FADD(se, FMUL(FMUL(ss(iz_u), ehb_u(id, iz_u)), (FSUB(z_u((iz_u + 1)), z_u(iz_u)))));
    // module_sf_bep.F:1625
    }
    // module_sf_bep.F:1626
    }
    // module_sf_bep.F:1628
    fact = FDIV(FDIV(FDIV(FDIV(bs(id), (FADD(ws(id), bs(id)))), vl(iz)), dz(iz)), float(nd));
    // module_sf_bep.F:1629
    b_t(iz) = FADD(b_t(iz), FMUL(st, fact));
    // module_sf_bep.F:1630
    b_u(iz) = FADD(b_u(iz), FMUL(su, fact));
    // module_sf_bep.F:1631
    b_v(iz) = FADD(b_v(iz), FMUL(sv, fact));
    // module_sf_bep.F:1632
    b_e(iz) = FADD(b_e(iz), FMUL(se, fact));
    // module_sf_bep.F:1633
    }
    // module_sf_bep.F:1634
    }
    // module_sf_bep.F:1638
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1639
    uet(iz) = 0.f;
    // module_sf_bep.F:1640
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1641
    vtb = 0.f;
    // module_sf_bep.F:1642
    vta = 0.f;
    // module_sf_bep.F:1643
    vua = 0.f;
    // module_sf_bep.F:1644
    vub = 0.f;
    // module_sf_bep.F:1645
    vva = 0.f;
    // module_sf_bep.F:1646
    vvb = 0.f;
    // module_sf_bep.F:1647
    veb = 0.f;
    // module_sf_bep.F:1648
    vte = 0.f;
    // module_sf_bep.F:1649
    for (iz_u = 1; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1650
    dzz = fmaxf(FSUB(fminf(z_u((iz_u + 1)), z((iz + 1))), fmaxf(z_u(iz_u), z(iz))), 0.f);
    // module_sf_bep.F:1651
    fact = FDIV(dzz, (FADD(ws(id), bs(id))));
    // module_sf_bep.F:1652
    vtb = FADD(vtb, FMUL(FMUL(pb((iz_u + 1)), (FADD(tvb_u(((2 * id) - 1), iz_u), tvb_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1654
    vta = FADD(vta, FMUL(FMUL(pb((iz_u + 1)), (FADD(tva_u(((2 * id) - 1), iz_u), tva_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1656
    vua = FADD(vua, FMUL(FMUL(pb((iz_u + 1)), (FADD(uva_u(((2 * id) - 1), iz_u), uva_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1658
    vva = FADD(vva, FMUL(FMUL(pb((iz_u + 1)), (FADD(vva_u(((2 * id) - 1), iz_u), vva_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1660
    vub = FADD(vub, FMUL(FMUL(pb((iz_u + 1)), (FADD(uvb_u(((2 * id) - 1), iz_u), uvb_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1662
    vvb = FADD(vvb, FMUL(FMUL(pb((iz_u + 1)), (FADD(vvb_u(((2 * id) - 1), iz_u), vvb_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1664
    veb = FADD(veb, FMUL(FMUL(pb((iz_u + 1)), (FADD(evb_u(((2 * id) - 1), iz_u), evb_u((2 * id), iz_u)))), fact));
    // module_sf_bep.F:1666
    }
    // module_sf_bep.F:1668
    fact = FDIV(FDIV(FDIV(1.f, vl(iz)), dz(iz)), float(nd));
    // module_sf_bep.F:1669
    b_t(iz) = FADD(b_t(iz), FMUL(vtb, fact));
    // module_sf_bep.F:1670
    a_t(iz) = FADD(a_t(iz), FMUL(vta, fact));
    // module_sf_bep.F:1671
    a_u(iz) = FADD(a_u(iz), FMUL(vua, fact));
    // module_sf_bep.F:1672
    a_v(iz) = FADD(a_v(iz), FMUL(vva, fact));
    // module_sf_bep.F:1673
    b_u(iz) = FADD(b_u(iz), FMUL(vub, fact));
    // module_sf_bep.F:1674
    b_v(iz) = FADD(b_v(iz), FMUL(vvb, fact));
    // module_sf_bep.F:1675
    b_e(iz) = FADD(b_e(iz), FMUL(veb, fact));
    // module_sf_bep.F:1676
    uet(iz) = FADD(uet(iz), FMUL(vte, fact));
    // module_sf_bep.F:1677
    }
    // module_sf_bep.F:1678
    }
    // module_sf_bep.F:1682
    return;
}

// module_sf_bep.F:1689-1769
__device__ void bep_interp_length(BepWS &arena, int &error, int nd, int kms, int kme, int kts, int kte, int nz_u, BepArray<float,1> z_u, BepArray<float,1> z, BepArray<float,1> ss, BepArray<float,1> ws, BepArray<float,1> bs, BepArray<float,1> dlg, BepArray<float,1> dl_u) {
    BepFrame frame(arena);
    float dlgtmp;
    int id;
    int iz;
    int iz_u;
    float sftot;
    float ulu;
    float ssl;
    // module_sf_bep.F:1731
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1732
    ulu = 0.f;
    // module_sf_bep.F:1733
    ssl = 0.f;
    // module_sf_bep.F:1734
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1735
    for (iz_u = 2; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1736
    if ((z_u(iz_u) > z(iz))) {
    // module_sf_bep.F:1737
    ulu = FADD(ulu, FDIV(FDIV(ss(iz_u), z_u(iz_u)), float(nd)));
    // module_sf_bep.F:1738
    ssl = FADD(ssl, FDIV(ss(iz_u), float(nd)));
    // module_sf_bep.F:1739
    }
    // module_sf_bep.F:1740
    }
    // module_sf_bep.F:1741
    }
    // module_sf_bep.F:1743
    if ((ulu != 0)) {
    // module_sf_bep.F:1744
    dl_u(iz) = FDIV(ssl, ulu);
    // module_sf_bep.F:1745
    } else {
    // module_sf_bep.F:1746
    dl_u(iz) = 0.f;
    // module_sf_bep.F:1747
    }
    // module_sf_bep.F:1748
    }
    // module_sf_bep.F:1751
    for (iz = kts; iz <= kte; iz += 1) {
    // module_sf_bep.F:1752
    dlg(iz) = 0.f;
    // module_sf_bep.F:1753
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1754
    sftot = ws(id);
    // module_sf_bep.F:1755
    dlgtmp = FDIV(ws(id), (FDIV((FADD(z(iz), z((iz + 1)))), 2.f)));
    // module_sf_bep.F:1756
    for (iz_u = 1; iz_u <= nz_u; iz_u += 1) {
    // module_sf_bep.F:1757
    if ((FDIV((FADD(z(iz), z((iz + 1)))), 2.f) > z_u(iz_u))) {
    // module_sf_bep.F:1758
    dlgtmp = FADD(dlgtmp, FDIV(FMUL(ss(iz_u), bs(id)), (FSUB(FDIV((FADD(z(iz), z((iz + 1)))), 2.f), z_u(iz_u)))));
    // module_sf_bep.F:1760
    sftot = FADD(sftot, FMUL(ss(iz_u), bs(id)));
    // module_sf_bep.F:1761
    }
    // module_sf_bep.F:1762
    }
    // module_sf_bep.F:1763
    dlg(iz) = FADD(dlg(iz), FDIV(FDIV(dlgtmp, sftot), float(nd)));
    // module_sf_bep.F:1764
    }
    // module_sf_bep.F:1765
    dlg(iz) = FDIV(1.f, dlg(iz));
    // module_sf_bep.F:1766
    }
    // module_sf_bep.F:1768
    return;
}

// module_sf_bep.F:1774-1891
__device__ void bep_shadow_mas(BepWS &arena, int &error, int nd, int nz_u, float zr, float deltar, float ah, BepArray<float,1> drst, BepArray<float,1> ws, BepArray<float,1> ss, BepArray<float,1> pb, BepArray<float,1> z, float rs, BepArray<float,2> rsw, BepArray<float,1> rsg) {
    BepFrame frame(arena);
    int id;
    int iz;
    int jz;
    float aae;
    float aaw;
    float bbb;
    float phix;
    float rd;
    float rtot;
    float wsd;
    // module_sf_bep.F:1815
    if (((rs == 0) || (glibc_sinf(zr) == 1))) {
    // module_sf_bep.F:1816
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1817
    rsg(id) = 0.f;
    // module_sf_bep.F:1818
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:1819
    rsw(((2 * id) - 1), iz) = 0.f;
    // module_sf_bep.F:1820
    rsw((2 * id), iz) = 0.f;
    // module_sf_bep.F:1821
    }
    // module_sf_bep.F:1822
    }
    // module_sf_bep.F:1823
    } else {
    // module_sf_bep.F:1825
    if ((fabsf(glibc_sinf(zr)) > 1.e-10f)) {
    // module_sf_bep.F:1826
    if ((FDIV(FMUL(glibc_cosf(deltar), glibc_sinf(ah)), glibc_sinf(zr)) >= 1)) {
    // module_sf_bep.F:1827
    bbb = FDIV(pi, 2.f);
    // module_sf_bep.F:1828
    } else if ((FDIV(FMUL(glibc_cosf(deltar), glibc_sinf(ah)), glibc_sinf(zr)) <= (-1))) {
    // module_sf_bep.F:1829
    bbb = FDIV((-pi), 2.f);
    // module_sf_bep.F:1830
    } else {
    // module_sf_bep.F:1831
    bbb = glibc_asinf(FDIV(FMUL(glibc_cosf(deltar), glibc_sinf(ah)), glibc_sinf(zr)));
    // module_sf_bep.F:1832
    }
    // module_sf_bep.F:1833
    } else {
    // module_sf_bep.F:1834
    if ((FMUL(glibc_cosf(deltar), glibc_sinf(ah)) >= 0)) {
    // module_sf_bep.F:1835
    bbb = FDIV(pi, 2.f);
    // module_sf_bep.F:1836
    } else if ((FMUL(glibc_cosf(deltar), glibc_sinf(ah)) < 0)) {
    // module_sf_bep.F:1837
    bbb = FDIV((-pi), 2.f);
    // module_sf_bep.F:1838
    }
    // module_sf_bep.F:1839
    }
    // module_sf_bep.F:1841
    phix = zr;
    // module_sf_bep.F:1843
    for (id = 1; id <= nd; id += 1) {
    // module_sf_bep.F:1845
    rsg(id) = 0.f;
    // module_sf_bep.F:1847
    aae = FSUB(bbb, drst(id));
    // module_sf_bep.F:1848
    aaw = FADD(FSUB(bbb, drst(id)), pi);
    // module_sf_bep.F:1850
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:1851
    rsw(((2 * id) - 1), iz) = 0.f;
    // module_sf_bep.F:1852
    rsw((2 * id), iz) = 0.f;
    // module_sf_bep.F:1853
    if ((pb((iz + 1)) > 0.f)) {
    // module_sf_bep.F:1854
    for (jz = 1; jz <= nz_u; jz += 1) {
    // module_sf_bep.F:1855
    if ((fabsf(glibc_sinf(aae)) > 1.e-10f)) {
    // module_sf_bep.F:1856
    bep_shade_wall(arena, error, z(iz), z((iz + 1)), z((jz + 1)), phix, aae, ws(id), rd); if (error) return;
    // module_sf_bep.F:1858
    rsw(((2 * id) - 1), iz) = FADD(rsw(((2 * id) - 1), iz), FDIV(FMUL(FMUL(rs, rd), ss((jz + 1))), pb((iz + 1))));
    // module_sf_bep.F:1859
    }
    // module_sf_bep.F:1861
    if ((fabsf(glibc_sinf(aaw)) > 1.e-10f)) {
    // module_sf_bep.F:1862
    bep_shade_wall(arena, error, z(iz), z((iz + 1)), z((jz + 1)), phix, aaw, ws(id), rd); if (error) return;
    // module_sf_bep.F:1864
    rsw((2 * id), iz) = FADD(rsw((2 * id), iz), FDIV(FMUL(FMUL(rs, rd), ss((jz + 1))), pb((iz + 1))));
    // module_sf_bep.F:1865
    }
    // module_sf_bep.F:1866
    }
    // module_sf_bep.F:1867
    }
    // module_sf_bep.F:1868
    }
    // module_sf_bep.F:1869
    if ((fabsf(glibc_sinf(aae)) > 1.e-10f)) {
    // module_sf_bep.F:1870
    wsd = fabsf(FDIV(ws(id), glibc_sinf(aae)));
    // module_sf_bep.F:1872
    for (jz = 1; jz <= nz_u; jz += 1) {
    // module_sf_bep.F:1873
    rd = fmaxf(0.f, FSUB(wsd, FMUL(z((jz + 1)), glibc_tanf(phix))));
    // module_sf_bep.F:1874
    rsg(id) = FADD(rsg(id), FDIV(FMUL(FMUL(rs, rd), ss((jz + 1))), wsd));
    // module_sf_bep.F:1875
    }
    // module_sf_bep.F:1876
    rtot = 0.f;
    // module_sf_bep.F:1878
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:1879
    rtot = FADD(rtot, FMUL((FADD(rsw((2 * id), iz), rsw(((2 * id) - 1), iz))), (FSUB(z((iz + 1)), z(iz)))));
    // module_sf_bep.F:1881
    }
    // module_sf_bep.F:1882
    rtot = FADD(rtot, FMUL(rsg(id), ws(id)));
    // module_sf_bep.F:1883
    } else {
    // module_sf_bep.F:1884
    rsg(id) = rs;
    // module_sf_bep.F:1885
    }
    // module_sf_bep.F:1887
    }
    // module_sf_bep.F:1888
    }
    // module_sf_bep.F:1890
    return;
}

// module_sf_bep.F:1896-1946
__device__ void bep_shade_wall(BepWS &arena, int &error, float z1, float z2, float hu, float phix, float aa, float ws, float &rd) {
    BepFrame frame(arena);
    float x1;
    float x2;
    // module_sf_bep.F:1939
    x1 = fminf(FMUL((FSUB(hu, z1)), glibc_tanf(phix)), fmaxf(0.f, FDIV(ws, glibc_sinf(aa))));
    // module_sf_bep.F:1941
    x2 = fmaxf(FMUL((FSUB(hu, z2)), glibc_tanf(phix)), 0.f);
    // module_sf_bep.F:1943
    rd = fmaxf(0.f, FDIV(FMUL(glibc_sinf(aa), (fmaxf(0.f, FSUB(x1, x2)))), (FSUB(z2, z1))));
    // module_sf_bep.F:1945
    return;
}

// module_sf_bep.F:1951-2092
__device__ void bep_long_rad(BepWS &arena, int &error, int iurb, int nz_u, int id, float emw, float emg, BepArray<float,3> fwg, BepArray<float,4> fww, BepArray<float,3> fgw, BepArray<float,3> fsw, BepArray<float,2> fsg, BepArray<float,2> tg, BepArray<float,3> tw, BepArray<float,1> rlg, BepArray<float,2> rlw, float rl, BepArray<float,1> pb) {
    BepFrame frame(arena);
    int i;
    int j;
    BepArray<float,2> aaa(arena.alloc<float>((((2 * nz_um) + 1)) * (((2 * nz_um) + 1))), 32, ((2 * nz_um) + 1), ((2 * nz_um) + 1), 1, 1);
    BepArray<float,1> bbb(arena.alloc<float>((((2 * nz_um) + 1))), 32, ((2 * nz_um) + 1), 1);
    // module_sf_bep.F:2006
    for (i = 1; i <= nz_u; i += 1) {
    // module_sf_bep.F:2008
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2009
    aaa(i, j) = 0.f;
    // module_sf_bep.F:2010
    }
    // module_sf_bep.F:2012
    aaa(i, i) = 1.f;
    // module_sf_bep.F:2014
    for (j = (nz_u + 1); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2015
    aaa(i, j) = FMUL(FMUL((-(FSUB(1.f, emw))), fww((j - nz_u), i, id, iurb)), pb(((j - nz_u) + 1)));
    // module_sf_bep.F:2016
    }
    // module_sf_bep.F:2019
    aaa(i, ((2 * nz_u) + 1)) = FMUL((-(FSUB(1.f, emg))), fgw(i, id, iurb));
    // module_sf_bep.F:2021
    bbb(i) = FADD(FMUL(fsw(i, id, iurb), rl), FMUL(FMUL(FMUL(emg, fgw(i, id, iurb)), sigma), FMUL(FMUL(tg(id, ng_u), tg(id, ng_u)), FMUL(tg(id, ng_u), tg(id, ng_u)))));
    // module_sf_bep.F:2022
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2023
    bbb(i) = FADD(FADD(bbb(i), FMUL(FMUL(FMUL(FMUL(pb((j + 1)), emw), sigma), fww(j, i, id, iurb)), FMUL(FMUL(tw((2 * id), j, nwr_u), tw((2 * id), j, nwr_u)), FMUL(tw((2 * id), j, nwr_u), tw((2 * id), j, nwr_u))))), FMUL(FMUL(fww(j, i, id, iurb), rl), (FSUB(1.f, pb((j + 1))))));
    // module_sf_bep.F:2026
    }
    // module_sf_bep.F:2028
    }
    // module_sf_bep.F:2032
    for (i = (1 + nz_u); i <= (2 * nz_u); i += 1) {
    // module_sf_bep.F:2034
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2035
    aaa(i, j) = FMUL(FMUL((-(FSUB(1.f, emw))), fww(j, (i - nz_u), id, iurb)), pb((j + 1)));
    // module_sf_bep.F:2036
    }
    // module_sf_bep.F:2038
    for (j = (1 + nz_u); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2039
    aaa(i, j) = 0.f;
    // module_sf_bep.F:2040
    }
    // module_sf_bep.F:2042
    aaa(i, i) = 1.f;
    // module_sf_bep.F:2045
    aaa(i, ((2 * nz_u) + 1)) = FMUL((-(FSUB(1.f, emg))), fgw((i - nz_u), id, iurb));
    // module_sf_bep.F:2047
    bbb(i) = FADD(FMUL(fsw((i - nz_u), id, iurb), rl), FMUL(FMUL(FMUL(emg, fgw((i - nz_u), id, iurb)), sigma), FMUL(FMUL(tg(id, ng_u), tg(id, ng_u)), FMUL(tg(id, ng_u), tg(id, ng_u)))));
    // module_sf_bep.F:2050
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2051
    bbb(i) = FADD(FADD(bbb(i), FMUL(FMUL(FMUL(FMUL(pb((j + 1)), emw), sigma), fww(j, (i - nz_u), id, iurb)), FMUL(FMUL(tw(((2 * id) - 1), j, nwr_u), tw(((2 * id) - 1), j, nwr_u)), FMUL(tw(((2 * id) - 1), j, nwr_u), tw(((2 * id) - 1), j, nwr_u))))), FMUL(FMUL(fww(j, (i - nz_u), id, iurb), rl), (FSUB(1.f, pb((j + 1))))));
    // module_sf_bep.F:2054
    }
    // module_sf_bep.F:2056
    }
    // module_sf_bep.F:2059
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2060
    aaa(((2 * nz_u) + 1), j) = FMUL(FMUL((-(FSUB(1.f, emw))), fwg(j, id, iurb)), pb((j + 1)));
    // module_sf_bep.F:2061
    }
    // module_sf_bep.F:2063
    for (j = (nz_u + 1); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2064
    aaa(((2 * nz_u) + 1), j) = FMUL(FMUL((-(FSUB(1.f, emw))), fwg((j - nz_u), id, iurb)), pb(((j - nz_u) + 1)));
    // module_sf_bep.F:2065
    }
    // module_sf_bep.F:2067
    aaa(((2 * nz_u) + 1), ((2 * nz_u) + 1)) = 1.f;
    // module_sf_bep.F:2069
    bbb(((2 * nz_u) + 1)) = FMUL(fsg(id, iurb), rl);
    // module_sf_bep.F:2071
    for (i = 1; i <= nz_u; i += 1) {
    // module_sf_bep.F:2072
    bbb(((2 * nz_u) + 1)) = FADD(FADD(bbb(((2 * nz_u) + 1)), FMUL(FMUL(FMUL(FMUL(emw, sigma), fwg(i, id, iurb)), pb((i + 1))), (FADD(FMUL(FMUL(tw(((2 * id) - 1), i, nwr_u), tw(((2 * id) - 1), i, nwr_u)), FMUL(tw(((2 * id) - 1), i, nwr_u), tw(((2 * id) - 1), i, nwr_u))), FMUL(FMUL(tw((2 * id), i, nwr_u), tw((2 * id), i, nwr_u)), FMUL(tw((2 * id), i, nwr_u), tw((2 * id), i, nwr_u))))))), FMUL(FMUL(FMUL(2.f, fwg(i, id, iurb)), (FSUB(1.f, pb((i + 1))))), rl));
    // module_sf_bep.F:2075
    }
    // module_sf_bep.F:2079
    bep_gaussj(arena, error, aaa, ((2 * nz_u) + 1), bbb, ((2 * nz_um) + 1)); if (error) return;
    // module_sf_bep.F:2081
    for (i = 1; i <= nz_u; i += 1) {
    // module_sf_bep.F:2082
    rlw(((2 * id) - 1), i) = bbb(i);
    // module_sf_bep.F:2083
    }
    // module_sf_bep.F:2085
    for (i = (nz_u + 1); i <= (2 * nz_u); i += 1) {
    // module_sf_bep.F:2086
    rlw((2 * id), (i - nz_u)) = bbb(i);
    // module_sf_bep.F:2087
    }
    // module_sf_bep.F:2089
    rlg(id) = bbb(((2 * nz_u) + 1));
    // module_sf_bep.F:2091
    return;
}

// module_sf_bep.F:2097-2205
__device__ void bep_short_rad(BepWS &arena, int &error, int iurb, int nz_u, int id, float albw, float albg, BepArray<float,3> fwg, BepArray<float,4> fww, BepArray<float,3> fgw, BepArray<float,1> rsg, BepArray<float,2> rsw, BepArray<float,1> pb) {
    BepFrame frame(arena);
    int i;
    int j;
    BepArray<float,2> aaa(arena.alloc<float>((((2 * nz_um) + 1)) * (((2 * nz_um) + 1))), 32, ((2 * nz_um) + 1), ((2 * nz_um) + 1), 1, 1);
    BepArray<float,1> bbb(arena.alloc<float>((((2 * nz_um) + 1))), 32, ((2 * nz_um) + 1), 1);
    // module_sf_bep.F:2146
    for (i = 1; i <= nz_u; i += 1) {
    // module_sf_bep.F:2147
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2148
    aaa(i, j) = 0.f;
    // module_sf_bep.F:2149
    }
    // module_sf_bep.F:2151
    aaa(i, i) = 1.f;
    // module_sf_bep.F:2153
    for (j = (nz_u + 1); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2154
    aaa(i, j) = FMUL(FMUL((-albw), fww((j - nz_u), i, id, iurb)), pb(((j - nz_u) + 1)));
    // module_sf_bep.F:2155
    }
    // module_sf_bep.F:2157
    aaa(i, ((2 * nz_u) + 1)) = FMUL((-albg), fgw(i, id, iurb));
    // module_sf_bep.F:2158
    bbb(i) = rsw(((2 * id) - 1), i);
    // module_sf_bep.F:2160
    }
    // module_sf_bep.F:2164
    for (i = (1 + nz_u); i <= (2 * nz_u); i += 1) {
    // module_sf_bep.F:2165
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2166
    aaa(i, j) = FMUL(FMUL((-albw), fww(j, (i - nz_u), id, iurb)), pb((j + 1)));
    // module_sf_bep.F:2167
    }
    // module_sf_bep.F:2169
    for (j = (1 + nz_u); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2170
    aaa(i, j) = 0.f;
    // module_sf_bep.F:2171
    }
    // module_sf_bep.F:2173
    aaa(i, i) = 1.f;
    // module_sf_bep.F:2174
    aaa(i, ((2 * nz_u) + 1)) = FMUL((-albg), fgw((i - nz_u), id, iurb));
    // module_sf_bep.F:2175
    bbb(i) = rsw((2 * id), (i - nz_u));
    // module_sf_bep.F:2177
    }
    // module_sf_bep.F:2181
    for (j = 1; j <= nz_u; j += 1) {
    // module_sf_bep.F:2182
    aaa(((2 * nz_u) + 1), j) = FMUL(FMUL((-albw), fwg(j, id, iurb)), pb((j + 1)));
    // module_sf_bep.F:2183
    }
    // module_sf_bep.F:2185
    for (j = (nz_u + 1); j <= (2 * nz_u); j += 1) {
    // module_sf_bep.F:2186
    aaa(((2 * nz_u) + 1), j) = FMUL(FMUL((-albw), fwg((j - nz_u), id, iurb)), pb(((j - nz_u) + 1)));
    // module_sf_bep.F:2187
    }
    // module_sf_bep.F:2189
    aaa(((2 * nz_u) + 1), ((2 * nz_u) + 1)) = 1.f;
    // module_sf_bep.F:2190
    bbb(((2 * nz_u) + 1)) = rsg(id);
    // module_sf_bep.F:2192
    bep_gaussj(arena, error, aaa, ((2 * nz_u) + 1), bbb, ((2 * nz_um) + 1)); if (error) return;
    // module_sf_bep.F:2194
    for (i = 1; i <= nz_u; i += 1) {
    // module_sf_bep.F:2195
    rsw(((2 * id) - 1), i) = bbb(i);
    // module_sf_bep.F:2196
    }
    // module_sf_bep.F:2198
    for (i = (nz_u + 1); i <= (2 * nz_u); i += 1) {
    // module_sf_bep.F:2199
    rsw((2 * id), (i - nz_u)) = bbb(i);
    // module_sf_bep.F:2200
    }
    // module_sf_bep.F:2202
    rsg(id) = bbb(((2 * nz_u) + 1));
    // module_sf_bep.F:2204
    return;
}

// module_sf_bep.F:2211-2313
__device__ void bep_gaussj(BepWS &arena, int &error, BepArray<float,2> a, int n, BepArray<float,1> b, int np) {
    BepFrame frame(arena);
    const int nmax = 150;
    float big;
    float dum;
    int i;
    int icol;
    int irow;
    int j;
    int k;
    int l;
    int ll;
    BepArray<int,1> ipiv(arena.alloc<int>((nmax)), 32, nmax, 1);
    float pivinv;
    // module_sf_bep.F:2250
    for (j = 1; j <= n; j += 1) {
    // module_sf_bep.F:2251
    ipiv(j) = 0.f;
    // module_sf_bep.F:2252
    }
    // module_sf_bep.F:2254
    for (i = 1; i <= n; i += 1) {
    // module_sf_bep.F:2255
    big = 0.f;
    // module_sf_bep.F:2256
    for (j = 1; j <= n; j += 1) {
    // module_sf_bep.F:2257
    if ((ipiv(j) != 1)) {
    // module_sf_bep.F:2258
    for (k = 1; k <= n; k += 1) {
    // module_sf_bep.F:2259
    if ((ipiv(k) == 0)) {
    // module_sf_bep.F:2260
    if ((fabsf(a(j, k)) >= big)) {
    // module_sf_bep.F:2261
    big = fabsf(a(j, k));
    // module_sf_bep.F:2262
    irow = j;
    // module_sf_bep.F:2263
    icol = k;
    // module_sf_bep.F:2264
    }
    // module_sf_bep.F:2265
    } else if ((ipiv(k) > 1)) {
    // module_sf_bep.F:2266
    { error = 2266; return; }
    // module_sf_bep.F:2267
    }
    // module_sf_bep.F:2268
    }
    // module_sf_bep.F:2269
    }
    // module_sf_bep.F:2270
    }
    // module_sf_bep.F:2272
    ipiv(icol) = (ipiv(icol) + 1);
    // module_sf_bep.F:2274
    if ((irow != icol)) {
    // module_sf_bep.F:2275
    for (l = 1; l <= n; l += 1) {
    // module_sf_bep.F:2276
    dum = a(irow, l);
    // module_sf_bep.F:2277
    a(irow, l) = a(icol, l);
    // module_sf_bep.F:2278
    a(icol, l) = dum;
    // module_sf_bep.F:2279
    }
    // module_sf_bep.F:2281
    dum = b(irow);
    // module_sf_bep.F:2282
    b(irow) = b(icol);
    // module_sf_bep.F:2283
    b(icol) = dum;
    // module_sf_bep.F:2285
    }
    // module_sf_bep.F:2287
    if ((a(icol, icol) == 0)) { { error = 2287; return; } }
    // module_sf_bep.F:2289
    pivinv = FDIV(1.f, a(icol, icol));
    // module_sf_bep.F:2290
    a(icol, icol) = 1;
    // module_sf_bep.F:2292
    for (l = 1; l <= n; l += 1) {
    // module_sf_bep.F:2293
    a(icol, l) = FMUL(a(icol, l), pivinv);
    // module_sf_bep.F:2294
    }
    // module_sf_bep.F:2296
    b(icol) = FMUL(b(icol), pivinv);
    // module_sf_bep.F:2298
    for (ll = 1; ll <= n; ll += 1) {
    // module_sf_bep.F:2299
    if ((ll != icol)) {
    // module_sf_bep.F:2300
    dum = a(ll, icol);
    // module_sf_bep.F:2301
    a(ll, icol) = 0.f;
    // module_sf_bep.F:2302
    for (l = 1; l <= n; l += 1) {
    // module_sf_bep.F:2303
    a(ll, l) = FSUB(a(ll, l), FMUL(a(icol, l), dum));
    // module_sf_bep.F:2304
    }
    // module_sf_bep.F:2306
    b(ll) = FSUB(b(ll), FMUL(b(icol), dum));
    // module_sf_bep.F:2308
    }
    // module_sf_bep.F:2309
    }
    // module_sf_bep.F:2310
    }
    // module_sf_bep.F:2312
    return;
}

// module_sf_bep.F:2318-2409
__device__ void bep_soil_temp(BepWS &arena, int &error, int nz, BepArray<float,1> dz, BepArray<float,1> temp, float &pt, BepArray<float,1> ala, BepArray<float,1> cs, float rs, float rl, float press, float dt, float em, float alb, float &rt, float sf, float &gf) {
    BepFrame frame(arena);
    int iz;
    BepArray<float,2> a(arena.alloc<float>((nz) * (3)), 32, nz, 3, 1, 1);
    float alpha;
    BepArray<float,1> c(arena.alloc<float>((nz)), 32, nz, 1);
    BepArray<float,1> cddz(arena.alloc<float>(((nz + 2))), 32, (nz + 2), 1);
    float tsig;
    // module_sf_bep.F:2371
    tsig = temp(nz);
    // module_sf_bep.F:2372
    alpha = FADD(FSUB(FADD(FMUL((FSUB(1.f, alb)), rs), FMUL(em, rl)), FMUL(FMUL(em, sigma), (FMUL(FMUL(tsig, tsig), FMUL(tsig, tsig))))), sf);
    // module_sf_bep.F:2375
    cddz(1) = FDIV(ala(1), dz(1));
    // module_sf_bep.F:2376
    for (iz = 2; iz <= nz; iz += 1) {
    // module_sf_bep.F:2377
    cddz(iz) = FDIV(FMUL(2.f, ala(iz)), (FADD(dz(iz), dz((iz - 1)))));
    // module_sf_bep.F:2378
    }
    // module_sf_bep.F:2381
    a(1, 1) = 0.f;
    // module_sf_bep.F:2382
    a(1, 2) = 1.f;
    // module_sf_bep.F:2383
    a(1, 3) = 0.f;
    // module_sf_bep.F:2384
    c(1) = temp(1);
    // module_sf_bep.F:2386
    for (iz = 2; iz <= (nz - 1); iz += 1) {
    // module_sf_bep.F:2387
    a(iz, 1) = FDIV(FMUL((-cddz(iz)), dt), dz(iz));
    // module_sf_bep.F:2388
    a(iz, 2) = FADD(float(1), FDIV(FMUL(dt, (FADD(cddz(iz), cddz((iz + 1))))), dz(iz)));
    // module_sf_bep.F:2389
    a(iz, 3) = FDIV(FMUL((-cddz((iz + 1))), dt), dz(iz));
    // module_sf_bep.F:2390
    c(iz) = temp(iz);
    // module_sf_bep.F:2391
    }
    // module_sf_bep.F:2393
    a(nz, 1) = FDIV(FMUL((-dt), cddz(nz)), dz(nz));
    // module_sf_bep.F:2394
    a(nz, 2) = FADD(1.f, FDIV(FMUL(dt, cddz(nz)), dz(nz)));
    // module_sf_bep.F:2395
    a(nz, 3) = 0.f;
    // module_sf_bep.F:2396
    c(nz) = FADD(temp(nz), FDIV(FDIV(FMUL(dt, alpha), cs(nz)), dz(nz)));
    // module_sf_bep.F:2399
    bep_invert(arena, error, nz, a, c, temp); if (error) return;
    // module_sf_bep.F:2402
    pt = FMUL(temp(nz), gfk_pow((FDIV(press, 1.e+5f)), ((-rcp_u))));
    // module_sf_bep.F:2404
    rt = FSUB(FADD(FMUL((FSUB(1.f, alb)), rs), FMUL(em, rl)), FMUL(FMUL(em, sigma), (FMUL(FMUL(tsig, tsig), FMUL(tsig, tsig)))));
    // module_sf_bep.F:2407
    gf = FADD(FSUB(FADD(FMUL((FSUB(1.f, alb)), rs), FMUL(em, rl)), FMUL(FMUL(em, sigma), (FMUL(FMUL(tsig, tsig), FMUL(tsig, tsig))))), sf);
    // module_sf_bep.F:2408
    return;
}

// module_sf_bep.F:2414-2460
__device__ void bep_invert(BepWS &arena, int &error, int n, BepArray<float,2> a, BepArray<float,1> c, BepArray<float,1> x) {
    BepFrame frame(arena);
    int i;
    // module_sf_bep.F:2446
    for (i = (n - 1); i >= 1; i += (-1)) {
    // module_sf_bep.F:2447
    c(i) = FSUB(c(i), FDIV(FMUL(a(i, 3), c((i + 1))), a((i + 1), 2)));
    // module_sf_bep.F:2448
    a(i, 2) = FSUB(a(i, 2), FDIV(FMUL(a(i, 3), a((i + 1), 1)), a((i + 1), 2)));
    // module_sf_bep.F:2449
    }
    // module_sf_bep.F:2451
    for (i = 2; i <= n; i += 1) {
    // module_sf_bep.F:2452
    c(i) = FSUB(c(i), FDIV(FMUL(a(i, 1), c((i - 1))), a((i - 1), 2)));
    // module_sf_bep.F:2453
    }
    // module_sf_bep.F:2455
    for (i = 1; i <= n; i += 1) {
    // module_sf_bep.F:2456
    x(i) = FDIV(c(i), a(i, 2));
    // module_sf_bep.F:2457
    }
    // module_sf_bep.F:2459
    return;
}

// module_sf_bep.F:2466-2538
__device__ void bep_flux_wall(BepWS &arena, int &error, float ua, float va, float pt, float da, float ptw, float &uva, float &vva, float &uvb, float &vvb, float &tva, float &tvb, float &evb, float drst, float dt) {
    BepFrame frame(arena);
    float hc;
    float u_ort;
    float vett;
    // module_sf_bep.F:2512
    vett = gfk_pow((FADD(FMUL(ua, ua), FMUL(va, va))), .5f);
    // module_sf_bep.F:2514
    u_ort = fabsf((FSUB(FMUL(glibc_cosf(drst), ua), FMUL(glibc_sinf(drst), va))));
    // module_sf_bep.F:2516
    uva = FMUL(FMUL(FDIV(FMUL((-cdrag), u_ort), 2.f), glibc_cosf(drst)), glibc_cosf(drst));
    // module_sf_bep.F:2517
    vva = FMUL(FMUL(FDIV(FMUL((-cdrag), u_ort), 2.f), glibc_sinf(drst)), glibc_sinf(drst));
    // module_sf_bep.F:2519
    uvb = FMUL(FMUL(FMUL(FDIV(FMUL(cdrag, u_ort), 2.f), glibc_sinf(drst)), glibc_cosf(drst)), va);
    // module_sf_bep.F:2520
    vvb = FMUL(FMUL(FMUL(FDIV(FMUL(cdrag, u_ort), 2.f), glibc_sinf(drst)), glibc_cosf(drst)), ua);
    // module_sf_bep.F:2522
    hc = FMUL(5.678f, (FADD(1.09f, FMUL(0.23f, (FDIV(vett, 0.3048f))))));
    // module_sf_bep.F:2524
    if ((hc > FDIV(FMUL(da, cp_u), dt))) {
    // module_sf_bep.F:2525
    hc = FDIV(FMUL(da, cp_u), dt);
    // module_sf_bep.F:2526
    }
    // module_sf_bep.F:2532
    tvb = FSUB(FDIV(FDIV(FMUL(hc, ptw), da), cp_u), FMUL(FDIV(FDIV(hc, da), cp_u), pt));
    // module_sf_bep.F:2533
    tva = 0.f;
    // module_sf_bep.F:2535
    evb = FDIV(FMUL(cdrag, (gfk_pow(fabsf(u_ort), 3.f))), 2.f);
    // module_sf_bep.F:2537
    return;
}

// module_sf_bep.F:2544-2663
__device__ void bep_flux_flat(BepWS &arena, int &error, float dz, float z0, float ua, float va, float pt, float pt0, float ptg, float &uhb, float &vhb, float &thb, float &ehb) {
    BepFrame frame(arena);
    float tva;
    float tvb;
    float aa;
    float al;
    float buu;
    float c;
    float fbuw;
    float fbpt;
    float fh;
    float fm;
    float ric;
    float tstar;
    float ustar;
    float utot;
    float wstar;
    float zz;
    const float b = 9.4f;
    const float cm = 7.4f;
    const float ch = 5.3f;
    const float rr = 0.74f;
    const float tol = .001f;
    // module_sf_bep.F:2610
    utot = gfk_pow((FADD(FMUL(ua, ua), FMUL(va, va))), .5f);
    // module_sf_bep.F:2617
    zz = FDIV(dz, 2.f);
    // module_sf_bep.F:2627
    utot = fmaxf(utot, 0.01f);
    // module_sf_bep.F:2629
    ric = FDIV(FMUL(FMUL(FMUL(2.f, g_u), zz), (FSUB(pt, ptg))), (FMUL((FADD(pt, ptg)), (FMUL(utot, utot)))));
    // module_sf_bep.F:2631
    aa = FDIV(vk, gfk_log(FDIV(zz, z0)));
    // module_sf_bep.F:2635
    if ((ric > 0)) {
    // module_sf_bep.F:2636
    fm = FDIV(float(1), FMUL((FADD(float(1), FMUL(FMUL(0.5f, b), ric))), (FADD(float(1), FMUL(FMUL(0.5f, b), ric)))));
    // module_sf_bep.F:2637
    fh = fm;
    // module_sf_bep.F:2638
    } else {
    // module_sf_bep.F:2639
    c = FMUL(FMUL(FMUL(FMUL(b, cm), aa), aa), gfk_pow((FDIV(zz, z0)), .5f));
    // module_sf_bep.F:2640
    fm = FSUB(float(1), FDIV(FMUL(b, ric), (FADD(float(1), FMUL(c, gfk_pow(((-ric)), .5f))))));
    // module_sf_bep.F:2641
    c = FDIV(FMUL(c, ch), cm);
    // module_sf_bep.F:2642
    fh = FSUB(float(1), FDIV(FMUL(b, ric), (FADD(float(1), FMUL(c, gfk_pow(((-ric)), .5f))))));
    // module_sf_bep.F:2643
    }
    // module_sf_bep.F:2645
    fbuw = FMUL(FMUL(FMUL(FMUL((-aa), aa), utot), utot), fm);
    // module_sf_bep.F:2646
    fbpt = FDIV(FMUL(FMUL(FMUL(FMUL((-aa), aa), utot), (FSUB(pt, ptg))), fh), rr);
    // module_sf_bep.F:2648
    ustar = gfk_pow(((-fbuw)), .5f);
    // module_sf_bep.F:2649
    tstar = FDIV((-fbpt), ustar);
    // module_sf_bep.F:2651
    al = FDIV((FMUL(FMUL(vk, g_u), tstar)), (FMUL(FMUL(pt, ustar), ustar)));
    // module_sf_bep.F:2653
    buu = FMUL(FMUL(FDIV((-g_u), pt0), ustar), tstar);
    // module_sf_bep.F:2655
    uhb = FDIV(FMUL(FMUL((-ustar), ustar), ua), utot);
    // module_sf_bep.F:2656
    vhb = FDIV(FMUL(FMUL((-ustar), ustar), va), utot);
    // module_sf_bep.F:2657
    thb = FMUL((-ustar), tstar);
    // module_sf_bep.F:2659
    ehb = buu;
    // module_sf_bep.F:2662
    return;
}

// module_sf_bep.F:2668-2770
__device__ void bep_icbep(BepWS &arena, int &error, BepArray<int,1> nd_u, BepArray<float,2> h_b, BepArray<float,2> d_b, BepArray<float,2> ss_u, BepArray<float,2> pb_u, BepArray<int,1> nz_u, BepArray<float,1> z_u) {
    BepFrame frame(arena);
    int iz_u;
    int id;
    int ilu;
    int iurb;
    float dtot;
    float hbmax;
    // module_sf_bep.F:2707
    for (int wi=0; wi<z_u.size(); ++wi) z_u[wi] = 0.f;
    // module_sf_bep.F:2708
    for (int wi=0; wi<nz_u.size(); ++wi) nz_u[wi] = 0;
    // module_sf_bep.F:2709
    for (int wi=0; wi<ss_u.size(); ++wi) ss_u[wi] = 0.f;
    // module_sf_bep.F:2710
    for (int wi=0; wi<pb_u.size(); ++wi) pb_u[wi] = 0.f;
    // module_sf_bep.F:2714
    z_u(1) = 0.f;
    // module_sf_bep.F:2716
    for (iz_u = 1; iz_u <= (nz_um - 1); iz_u += 1) {
    // module_sf_bep.F:2717
    z_u((iz_u + 1)) = FADD(z_u(iz_u), dz_u);
    // module_sf_bep.F:2718
    }
    // module_sf_bep.F:2722
    for (iurb = 1; iurb <= nurbm; iurb += 1) {
    // module_sf_bep.F:2723
    dtot = 0.f;
    // module_sf_bep.F:2724
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:2725
    dtot = FADD(dtot, d_b(ilu, iurb));
    // module_sf_bep.F:2726
    }
    // module_sf_bep.F:2727
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:2728
    d_b(ilu, iurb) = FDIV(d_b(ilu, iurb), dtot);
    // module_sf_bep.F:2729
    }
    // module_sf_bep.F:2730
    }
    // module_sf_bep.F:2734
    for (iurb = 1; iurb <= nurbm; iurb += 1) {
    // module_sf_bep.F:2735
    hbmax = 0.f;
    // module_sf_bep.F:2736
    nz_u(iurb) = 0;
    // module_sf_bep.F:2737
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:2738
    if ((h_b(ilu, iurb) > hbmax)) { hbmax = h_b(ilu, iurb); }
    // module_sf_bep.F:2739
    }
    // module_sf_bep.F:2741
    for (iz_u = 1; iz_u <= (nz_um - 1); iz_u += 1) {
    // module_sf_bep.F:2742
    if ((z_u((iz_u + 1)) > hbmax)) { goto L10; }
    // module_sf_bep.F:2743
    }
    // module_sf_bep.F:2745
    L10: ;
    // module_sf_bep.F:2746
    nz_u(iurb) = (iz_u + 1);
    // module_sf_bep.F:2740-2757 has no check before z_u(nzu+1).
    // Guard undefined Fortran out-of-bounds table heights with a line flag.
    if(nz_u(iurb)+1>nz_um) {error=2740;return;}
    // module_sf_bep.F:2748
    for (id = 1; id <= nd_u(iurb); id += 1) {
    // module_sf_bep.F:2750
    for (iz_u = 1; iz_u <= nz_u(iurb); iz_u += 1) {
    // module_sf_bep.F:2751
    ss_u(iz_u, iurb) = 0.f;
    // module_sf_bep.F:2752
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:2753
    if (((z_u(iz_u) <= h_b(ilu, iurb)) && (z_u((iz_u + 1)) > h_b(ilu, iurb)))) {
    // module_sf_bep.F:2755
    ss_u(iz_u, iurb) = FADD(ss_u(iz_u, iurb), d_b(ilu, iurb));
    // module_sf_bep.F:2756
    }
    // module_sf_bep.F:2757
    }
    // module_sf_bep.F:2758
    }
    // module_sf_bep.F:2760
    pb_u(1, iurb) = 1.f;
    // module_sf_bep.F:2761
    for (iz_u = 1; iz_u <= nz_u(iurb); iz_u += 1) {
    // module_sf_bep.F:2762
    pb_u((iz_u + 1), iurb) = fmaxf(0.f, FSUB(pb_u(iz_u, iurb), ss_u(iz_u, iurb)));
    // module_sf_bep.F:2763
    }
    // module_sf_bep.F:2765
    }
    // module_sf_bep.F:2766
    }
    // module_sf_bep.F:2769
    return;
}

// module_sf_bep.F:2775-2934
__device__ void bep_view_factors(BepWS &arena, int &error, int iurb, int nz_u, int id, float dxy, BepArray<float,1> z, float ws, BepArray<float,4> fww, BepArray<float,3> fwg, BepArray<float,3> fgw, BepArray<float,2> fsg, BepArray<float,3> fsw, BepArray<float,3> fws) {
    BepFrame frame(arena);
    int jz;
    int iz;
    float hut;
    float f1;
    float f2;
    float f12;
    float f23;
    float f123;
    float ftot;
    float fprl;
    float fnrm;
    float a1;
    float a2;
    float a3;
    float a4;
    float a12;
    float a23;
    float a123;
    // module_sf_bep.F:2824
    hut = z((nz_u + 1));
    // module_sf_bep.F:2826
    for (jz = 1; jz <= nz_u; jz += 1) {
    // module_sf_bep.F:2830
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:2832
    bep_fprls(arena, error, fprl, dxy, fabsf(FSUB(z((jz + 1)), z(iz))), ws); if (error) return;
    // module_sf_bep.F:2833
    f123 = fprl;
    // module_sf_bep.F:2834
    bep_fprls(arena, error, fprl, dxy, fabsf(FSUB(z((jz + 1)), z((iz + 1)))), ws); if (error) return;
    // module_sf_bep.F:2835
    f23 = fprl;
    // module_sf_bep.F:2836
    bep_fprls(arena, error, fprl, dxy, fabsf(FSUB(z(jz), z(iz))), ws); if (error) return;
    // module_sf_bep.F:2837
    f12 = fprl;
    // module_sf_bep.F:2838
    bep_fprls(arena, error, fprl, dxy, fabsf(FSUB(z(jz), z((iz + 1)))), ws); if (error) return;
    // module_sf_bep.F:2839
    f2 = fprl;
    // module_sf_bep.F:2841
    a123 = FMUL(dxy, (fabsf(FSUB(z((jz + 1)), z(iz)))));
    // module_sf_bep.F:2842
    a12 = FMUL(dxy, (fabsf(FSUB(z(jz), z(iz)))));
    // module_sf_bep.F:2843
    a23 = FMUL(dxy, (fabsf(FSUB(z((jz + 1)), z((iz + 1))))));
    // module_sf_bep.F:2844
    a1 = FMUL(dxy, (fabsf(FSUB(z((iz + 1)), z(iz)))));
    // module_sf_bep.F:2845
    a2 = FMUL(dxy, (fabsf(FSUB(z(jz), z((iz + 1))))));
    // module_sf_bep.F:2846
    a3 = FMUL(dxy, (fabsf(FSUB(z((jz + 1)), z(jz)))));
    // module_sf_bep.F:2848
    ftot = FDIV(FMUL(0.5f, (FADD(FSUB(FSUB(FMUL(a123, f123), FMUL(a23, f23)), FMUL(a12, f12)), FMUL(a2, f2)))), a1);
    // module_sf_bep.F:2850
    fww(iz, jz, id, iurb) = FDIV(FMUL(ftot, a1), a3);
    // module_sf_bep.F:2852
    }
    // module_sf_bep.F:2856
    bep_fnrms(arena, error, fnrm, z((jz + 1)), dxy, ws); if (error) return;
    // module_sf_bep.F:2857
    f12 = fnrm;
    // module_sf_bep.F:2858
    bep_fnrms(arena, error, fnrm, z(jz), dxy, ws); if (error) return;
    // module_sf_bep.F:2859
    f1 = fnrm;
    // module_sf_bep.F:2861
    a1 = FMUL(ws, dxy);
    // module_sf_bep.F:2863
    a12 = FMUL(ws, dxy);
    // module_sf_bep.F:2865
    a4 = FMUL((FSUB(z((jz + 1)), z(jz))), dxy);
    // module_sf_bep.F:2867
    ftot = FDIV((FSUB(FMUL(a12, f12), FMUL(a12, f1))), a1);
    // module_sf_bep.F:2869
    fgw(jz, id, iurb) = FDIV(FMUL(ftot, a1), a4);
    // module_sf_bep.F:2873
    bep_fnrms(arena, error, fnrm, FSUB(hut, z(jz)), dxy, ws); if (error) return;
    // module_sf_bep.F:2874
    f12 = fnrm;
    // module_sf_bep.F:2875
    bep_fnrms(arena, error, fnrm, FSUB(hut, z((jz + 1))), dxy, ws); if (error) return;
    // module_sf_bep.F:2876
    f1 = fnrm;
    // module_sf_bep.F:2878
    a1 = FMUL(ws, dxy);
    // module_sf_bep.F:2880
    a12 = FMUL(ws, dxy);
    // module_sf_bep.F:2882
    a4 = FMUL((FSUB(z((jz + 1)), z(jz))), dxy);
    // module_sf_bep.F:2884
    ftot = FDIV((FSUB(FMUL(a12, f12), FMUL(a12, f1))), a1);
    // module_sf_bep.F:2886
    fsw(jz, id, iurb) = FDIV(FMUL(ftot, a1), a4);
    // module_sf_bep.F:2888
    }
    // module_sf_bep.F:2891
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:2892
    bep_fnrms(arena, error, fnrm, ws, dxy, FSUB(hut, z(iz))); if (error) return;
    // module_sf_bep.F:2893
    f12 = fnrm;
    // module_sf_bep.F:2894
    bep_fnrms(arena, error, fnrm, ws, dxy, FSUB(hut, z((iz + 1)))); if (error) return;
    // module_sf_bep.F:2895
    f1 = fnrm;
    // module_sf_bep.F:2896
    a1 = FMUL((FSUB(z((iz + 1)), z(iz))), dxy);
    // module_sf_bep.F:2897
    a2 = FMUL((FSUB(hut, z((iz + 1)))), dxy);
    // module_sf_bep.F:2898
    a12 = FMUL((FSUB(hut, z(iz))), dxy);
    // module_sf_bep.F:2899
    a4 = FMUL(ws, dxy);
    // module_sf_bep.F:2900
    ftot = FDIV((FSUB(FMUL(a12, f12), FMUL(a2, f1))), a1);
    // module_sf_bep.F:2901
    fws(iz, id, iurb) = FDIV(FMUL(ftot, a1), a4);
    // module_sf_bep.F:2903
    }
    // module_sf_bep.F:2907
    for (iz = 1; iz <= nz_u; iz += 1) {
    // module_sf_bep.F:2911
    bep_fnrms(arena, error, fnrm, ws, dxy, z((iz + 1))); if (error) return;
    // module_sf_bep.F:2912
    f12 = fnrm;
    // module_sf_bep.F:2913
    bep_fnrms(arena, error, fnrm, ws, dxy, z(iz)); if (error) return;
    // module_sf_bep.F:2914
    f1 = fnrm;
    // module_sf_bep.F:2916
    a1 = FMUL((FSUB(z((iz + 1)), z(iz))), dxy);
    // module_sf_bep.F:2918
    a2 = FMUL(z(iz), dxy);
    // module_sf_bep.F:2919
    a12 = FMUL(z((iz + 1)), dxy);
    // module_sf_bep.F:2920
    a4 = FMUL(ws, dxy);
    // module_sf_bep.F:2922
    ftot = FDIV((FSUB(FMUL(a12, f12), FMUL(a2, f1))), a1);
    // module_sf_bep.F:2924
    fwg(iz, id, iurb) = FDIV(FMUL(ftot, a1), a4);
    // module_sf_bep.F:2926
    }
    // module_sf_bep.F:2930
    bep_fprls(arena, error, fprl, dxy, ws, hut); if (error) return;
    // module_sf_bep.F:2931
    fsg(id, iurb) = fprl;
    // module_sf_bep.F:2933
    return;
}

// module_sf_bep.F:2939-2964
__device__ void bep_fprls(BepWS &arena, int &error, float &fprl, float a, float b, float c) {
    BepFrame frame(arena);
    float x;
    float y;
    // module_sf_bep.F:2950
    x = FDIV(a, c);
    // module_sf_bep.F:2951
    y = FDIV(b, c);
    // module_sf_bep.F:2953
    if (((a == 0) || (b == 0.f))) {
    // module_sf_bep.F:2954
    fprl = 0.f;
    // module_sf_bep.F:2955
    } else {
    // module_sf_bep.F:2956
    fprl = FSUB(FSUB(FADD(FADD(gfk_log(gfk_pow((FDIV(FMUL((FADD(1.f, FMUL(x, x))), (FADD(1.f, FMUL(y, y)))), (FADD(FADD(1.f, FMUL(x, x)), FMUL(y, y))))), .5f)), FMUL(FMUL(y, (gfk_pow((FADD(1.f, FMUL(x, x))), .5f))), glibc_atanf(FDIV(y, (gfk_pow((FADD(1.f, FMUL(x, x))), .5f)))))), FMUL(FMUL(x, (gfk_pow((FADD(1.f, FMUL(y, y))), .5f))), glibc_atanf(FDIV(x, (gfk_pow((FADD(1.f, FMUL(y, y))), .5f)))))), FMUL(y, glibc_atanf(y))), FMUL(x, glibc_atanf(x)));
    // module_sf_bep.F:2960
    fprl = FDIV(FMUL(fprl, 2.f), (FMUL(FMUL(pi, x), y)));
    // module_sf_bep.F:2961
    }
    // module_sf_bep.F:2963
    return;
}

// module_sf_bep.F:2969-2997
__device__ void bep_fnrms(BepWS &arena, int &error, float &fnrm, float a, float b, float c) {
    BepFrame frame(arena);
    float x;
    float y;
    float z;
    float a1;
    float a2;
    float a3;
    float a4;
    float a5;
    float a6;
    // module_sf_bep.F:2979
    x = FDIV(a, b);
    // module_sf_bep.F:2980
    y = FDIV(c, b);
    // module_sf_bep.F:2981
    z = FADD(FMUL(x, x), FMUL(y, y));
    // module_sf_bep.F:2983
    if (((y == 0) || (x == 0))) {
    // module_sf_bep.F:2984
    fnrm = 0.f;
    // module_sf_bep.F:2985
    } else {
    // module_sf_bep.F:2986
    a1 = gfk_log(FDIV(FMUL((FADD(1.f, FMUL(x, x))), (FADD(1.f, FMUL(y, y)))), (FADD(1.f, z))));
    // module_sf_bep.F:2987
    a2 = FMUL(FMUL(y, y), gfk_log(FDIV(FDIV(FMUL(FMUL(y, y), (FADD(1.f, z))), z), (FADD(1.f, FMUL(y, y))))));
    // module_sf_bep.F:2988
    a3 = FMUL(FMUL(x, x), gfk_log(FDIV(FDIV(FMUL(FMUL(x, x), (FADD(1.f, z))), z), (FADD(1.f, FMUL(x, x))))));
    // module_sf_bep.F:2989
    a4 = FMUL(y, glibc_atanf(FDIV(1.f, y)));
    // module_sf_bep.F:2990
    a5 = FMUL(x, glibc_atanf(FDIV(1.f, x)));
    // module_sf_bep.F:2991
    a6 = FMUL(sqrtf(z), glibc_atanf(FDIV(1.f, sqrtf(z))));
    // module_sf_bep.F:2992
    fnrm = FSUB(FADD(FADD(FMUL(0.25f, (FADD(FADD(a1, a2), a3))), a4), a5), a6);
    // module_sf_bep.F:2993
    fnrm = FDIV(fnrm, (FMUL(pi, y)));
    // module_sf_bep.F:2994
    }
    // module_sf_bep.F:2996
    return;
}

// module_sf_bep.F:3000-3108
__device__ void bep_init_para(BepWS &arena, int &error, const UrbanBepTable &tbl, BepArray<float,1> alag_u, BepArray<float,1> alaw_u, BepArray<float,1> alar_u, BepArray<float,1> csg_u, BepArray<float,1> csw_u, BepArray<float,1> csr_u, BepArray<float,1> twini_u, BepArray<float,1> trini_u, BepArray<float,1> tgini_u, BepArray<float,1> albg_u, BepArray<float,1> albw_u, BepArray<float,1> albr_u, BepArray<float,1> emg_u, BepArray<float,1> emw_u, BepArray<float,1> emr_u, BepArray<float,1> z0g_u, BepArray<float,1> z0r_u, BepArray<int,1> nd_u, BepArray<float,2> strd_u, BepArray<float,2> drst_u, BepArray<float,2> ws_u, BepArray<float,2> bs_u, BepArray<float,2> h_b, BepArray<float,2> d_b) {
    BepFrame frame(arena);
    int iurb;
    int i;
    int iu;
    int nurb;
    const int icate = tbl.icate;
    BepArray<float,1> capb_tbl(const_cast<float *>(tbl.capb_tbl), 1, nurbm, 1);
    BepArray<float,1> capr_tbl(const_cast<float *>(tbl.capr_tbl), 1, nurbm, 1);
    BepArray<float,1> capg_tbl(const_cast<float *>(tbl.capg_tbl), 1, nurbm, 1);
    BepArray<float,1> aksb_tbl(const_cast<float *>(tbl.aksb_tbl), 1, nurbm, 1);
    BepArray<float,1> aksr_tbl(const_cast<float *>(tbl.aksr_tbl), 1, nurbm, 1);
    BepArray<float,1> aksg_tbl(const_cast<float *>(tbl.aksg_tbl), 1, nurbm, 1);
    BepArray<float,1> tblend_tbl(const_cast<float *>(tbl.tblend_tbl), 1, nurbm, 1);
    BepArray<float,1> trlend_tbl(const_cast<float *>(tbl.trlend_tbl), 1, nurbm, 1);
    BepArray<float,1> tglend_tbl(const_cast<float *>(tbl.tglend_tbl), 1, nurbm, 1);
    BepArray<float,1> albb_tbl(const_cast<float *>(tbl.albb_tbl), 1, nurbm, 1);
    BepArray<float,1> albr_tbl(const_cast<float *>(tbl.albr_tbl), 1, nurbm, 1);
    BepArray<float,1> albg_tbl(const_cast<float *>(tbl.albg_tbl), 1, nurbm, 1);
    BepArray<float,1> epsb_tbl(const_cast<float *>(tbl.epsb_tbl), 1, nurbm, 1);
    BepArray<float,1> epsr_tbl(const_cast<float *>(tbl.epsr_tbl), 1, nurbm, 1);
    BepArray<float,1> epsg_tbl(const_cast<float *>(tbl.epsg_tbl), 1, nurbm, 1);
    BepArray<float,1> z0r_tbl(const_cast<float *>(tbl.z0r_tbl), 1, nurbm, 1);
    BepArray<float,1> z0g_tbl(const_cast<float *>(tbl.z0g_tbl), 1, nurbm, 1);
    BepArray<int,1> numdir_tbl(const_cast<int *>(tbl.numdir_tbl), 1, nurbm, 1);
    BepArray<float,2> street_direction_tbl(const_cast<float *>(tbl.street_direction_tbl), 1, 3, nurbm, 1, 1);
    BepArray<float,2> street_width_tbl(const_cast<float *>(tbl.street_width_tbl), 1, 3, nurbm, 1, 1);
    BepArray<float,2> building_width_tbl(const_cast<float *>(tbl.building_width_tbl), 1, 3, nurbm, 1, 1);
    BepArray<int,1> numhgt_tbl(const_cast<int *>(tbl.numhgt_tbl), 1, nurbm, 1);
    BepArray<float,2> height_bin_tbl(const_cast<float *>(tbl.height_bin_tbl), 1, 50, nurbm, 1, 1);
    BepArray<float,2> hpercent_bin_tbl(const_cast<float *>(tbl.hpercent_bin_tbl), 1, 50, nurbm, 1, 1);
    // module_sf_bep.F:3049
    for (int wi=0; wi<h_b.size(); ++wi) h_b[wi] = 0.f;
    // module_sf_bep.F:3050
    for (int wi=0; wi<d_b.size(); ++wi) d_b[wi] = 0.f;
    // module_sf_bep.F:3052
    nurb = icate;
    // module_sf_bep.F:3053
    for (iu = 1; iu <= nurb; iu += 1) {
    // module_sf_bep.F:3054
    nd_u(iu) = 0;
    // module_sf_bep.F:3055
    }
    // module_sf_bep.F:3057
    for (int wi=0; wi<csw_u.size(); ++wi) csw_u[wi] = FDIV(capb_tbl[wi], (FMUL((FDIV(1.0f, 4.1868f)), 1.e-6f)));
    // module_sf_bep.F:3058
    for (int wi=0; wi<csr_u.size(); ++wi) csr_u[wi] = FDIV(capr_tbl[wi], (FMUL((FDIV(1.0f, 4.1868f)), 1.e-6f)));
    // module_sf_bep.F:3059
    for (int wi=0; wi<csg_u.size(); ++wi) csg_u[wi] = FDIV(capg_tbl[wi], (FMUL((FDIV(1.0f, 4.1868f)), 1.e-6f)));
    // module_sf_bep.F:3060
    for (i = 1; i <= icate; i += 1) {
    // module_sf_bep.F:3061
    alaw_u(i) = FDIV(FDIV(aksb_tbl(i), csw_u(i)), (FMUL((FDIV(1.0f, 4.1868f)), 1.e-2f)));
    // module_sf_bep.F:3062
    alar_u(i) = FDIV(FDIV(aksr_tbl(i), csr_u(i)), (FMUL((FDIV(1.0f, 4.1868f)), 1.e-2f)));
    // module_sf_bep.F:3063
    alag_u(i) = FDIV(FDIV(aksg_tbl(i), csg_u(i)), (FMUL((FDIV(1.0f, 4.1868f)), 1.e-2f)));
    // module_sf_bep.F:3064
    }
    // module_sf_bep.F:3065
    for (int wi=0; wi<twini_u.size(); ++wi) twini_u[wi] = tblend_tbl[wi];
    // module_sf_bep.F:3066
    for (int wi=0; wi<trini_u.size(); ++wi) trini_u[wi] = trlend_tbl[wi];
    // module_sf_bep.F:3067
    for (int wi=0; wi<tgini_u.size(); ++wi) tgini_u[wi] = tglend_tbl[wi];
    // module_sf_bep.F:3068
    for (int wi=0; wi<albw_u.size(); ++wi) albw_u[wi] = albb_tbl[wi];
    // module_sf_bep.F:3069
    for (int wi=0; wi<albr_u.size(); ++wi) albr_u[wi] = albr_tbl[wi];
    // module_sf_bep.F:3070
    for (int wi=0; wi<albg_u.size(); ++wi) albg_u[wi] = albg_tbl[wi];
    // module_sf_bep.F:3071
    for (int wi=0; wi<emw_u.size(); ++wi) emw_u[wi] = epsb_tbl[wi];
    // module_sf_bep.F:3072
    for (int wi=0; wi<emr_u.size(); ++wi) emr_u[wi] = epsr_tbl[wi];
    // module_sf_bep.F:3073
    for (int wi=0; wi<emg_u.size(); ++wi) emg_u[wi] = epsg_tbl[wi];
    // module_sf_bep.F:3074
    for (int wi=0; wi<z0r_u.size(); ++wi) z0r_u[wi] = z0r_tbl[wi];
    // module_sf_bep.F:3075
    for (int wi=0; wi<z0g_u.size(); ++wi) z0g_u[wi] = z0g_tbl[wi];
    // module_sf_bep.F:3076
    for (int wi=0; wi<nd_u.size(); ++wi) nd_u[wi] = numdir_tbl[wi];
    // module_sf_bep.F:3077
    for (iu = 1; iu <= icate; iu += 1) {
    // module_sf_bep.F:3078
    if ((ndm < nd_u(iu))) {
    // module_sf_bep.F:3079
    ; // diagnostic omitted
    // module_sf_bep.F:3080
    ; // diagnostic omitted
    // module_sf_bep.F:3081
    { error = 3081; return; }
    // module_sf_bep.F:3082
    }
    // module_sf_bep.F:3083
    for (i = 1; i <= nd_u(iu); i += 1) {
    // module_sf_bep.F:3084
    drst_u(i, iu) = FDIV(FMUL(street_direction_tbl(i, iu), pi), 180.f);
    // module_sf_bep.F:3085
    ws_u(i, iu) = street_width_tbl(i, iu);
    // module_sf_bep.F:3086
    bs_u(i, iu) = building_width_tbl(i, iu);
    // module_sf_bep.F:3087
    }
    // module_sf_bep.F:3088
    }
    // module_sf_bep.F:3089
    for (iu = 1; iu <= icate; iu += 1) {
    // module_sf_bep.F:3090
    if ((nz_um < (numhgt_tbl(iu) + 3))) {
    // module_sf_bep.F:3091
    ; // diagnostic omitted
    // module_sf_bep.F:3092
    ; // diagnostic omitted
    // module_sf_bep.F:3093
    { error = 3093; return; }
    // module_sf_bep.F:3094
    }
    // module_sf_bep.F:3095
    for (i = 1; i <= numhgt_tbl(iu); i += 1) {
    // module_sf_bep.F:3096
    h_b(i, iu) = height_bin_tbl(i, iu);
    // module_sf_bep.F:3097
    d_b(i, iu) = hpercent_bin_tbl(i, iu);
    // module_sf_bep.F:3098
    }
    // module_sf_bep.F:3099
    }
    // module_sf_bep.F:3101
    for (i = 1; i <= ndm; i += 1) {
    // module_sf_bep.F:3102
    for (iu = 1; iu <= nurbm; iu += 1) {
    // module_sf_bep.F:3103
    strd_u(i, iu) = 100000.f;
    // module_sf_bep.F:3104
    }
    // module_sf_bep.F:3105
    }
    // module_sf_bep.F:3107
    return;
}

// module_sf_bep.F:3194-3314
__device__ void bep_upward_rad(BepWS &arena, int &error, int ndu, int nzu, BepArray<float,1> ws, BepArray<float,1> bs, float sigma, BepArray<float,1> pb, BepArray<float,1> ss, BepArray<float,2> tg, float emg_u, float albg_u, BepArray<float,1> rlg, BepArray<float,1> rsg, BepArray<float,1> sfg, BepArray<float,3> tw, float emw_u, float albw_u, BepArray<float,2> rlw, BepArray<float,2> rsw, BepArray<float,2> sfw, BepArray<float,3> tr, float emr_u, float albr_u, float rld, float rs, BepArray<float,2> sfr, float &rs_abs, float &rl_up, float &emiss, float &grdflx_urb) {
    BepFrame frame(arena);
    int id;
    int iz;
    int iw;
    float rl_inc;
    float rl_emit;
    float gfl;
    int ix;
    int iy;
    int iwrong;
    // module_sf_bep.F:3246
    iwrong = 1;
    // module_sf_bep.F:3247
    for (iz = 1; iz <= (nzu + 1); iz += 1) {
    // module_sf_bep.F:3248
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:3249
    for (iw = 1; iw <= nwr_u; iw += 1) {
    // module_sf_bep.F:3250
    if ((tr(id, iz, iw) < 100.f)) {
    // module_sf_bep.F:3251
    ; // diagnostic omitted
    // module_sf_bep.F:3252
    iwrong = 0;
    // module_sf_bep.F:3253
    }
    // module_sf_bep.F:3254
    if ((tw(((2 * id) - 1), iz, iw) < 100.f)) {
    // module_sf_bep.F:3255
    ; // diagnostic omitted
    // module_sf_bep.F:3256
    iwrong = 0;
    // module_sf_bep.F:3257
    }
    // module_sf_bep.F:3258
    if ((tw((2 * id), iz, iw) < 100.f)) {
    // module_sf_bep.F:3259
    ; // diagnostic omitted
    // module_sf_bep.F:3260
    iwrong = 0;
    // module_sf_bep.F:3261
    }
    // module_sf_bep.F:3262
    }
    // module_sf_bep.F:3263
    }
    // module_sf_bep.F:3264
    }
    // module_sf_bep.F:3265
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:3266
    for (iw = 1; iw <= ng_u; iw += 1) {
    // module_sf_bep.F:3267
    if ((tg(id, iw) < 100.f)) {
    // module_sf_bep.F:3268
    ; // diagnostic omitted
    // module_sf_bep.F:3269
    iwrong = 0;
    // module_sf_bep.F:3270
    }
    // module_sf_bep.F:3271
    }
    // module_sf_bep.F:3272
    }
    // module_sf_bep.F:3273
    if ((iwrong == 0)) { { error = 3273; return; } }
    // module_sf_bep.F:3275
    rl_up = 0.f;
    // module_sf_bep.F:3277
    rs_abs = 0.f;
    // module_sf_bep.F:3278
    rl_inc = 0.f;
    // module_sf_bep.F:3279
    emiss = 0.f;
    // module_sf_bep.F:3280
    rl_emit = 0.f;
    // module_sf_bep.F:3281
    grdflx_urb = 0.f;
    // module_sf_bep.F:3282
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:3283
    rl_emit = FSUB(rl_emit, FDIV(FDIV(FMUL((FADD(FMUL(FMUL(emg_u, sigma), (gfk_pow(tg(id, ng_u), 4.f))), FMUL((FSUB(float(1), emg_u)), rlg(id)))), ws(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3284
    rl_inc = FADD(rl_inc, FDIV(FDIV(FMUL(rlg(id), ws(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3285
    rs_abs = FADD(rs_abs, FDIV(FDIV(FMUL(FMUL((FSUB(1.f, albg_u)), rsg(id)), ws(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3286
    gfl = FADD(FSUB(FADD(FMUL((FSUB(1.f, albg_u)), rsg(id)), FMUL(emg_u, rlg(id))), FMUL(FMUL(emg_u, sigma), (gfk_pow(tg(id, ng_u), 4.f)))), sfg(id));
    // module_sf_bep.F:3287
    grdflx_urb = FSUB(grdflx_urb, FDIV(FDIV(FMUL(gfl, ws(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3289
    for (iz = 2; iz <= nzu; iz += 1) {
    // module_sf_bep.F:3290
    rl_emit = FSUB(rl_emit, FDIV(FDIV(FMUL(FMUL((FADD(FMUL(FMUL(emr_u, sigma), (gfk_pow(tr(id, iz, nwr_u), 4.f))), FMUL((FSUB(float(1), emr_u)), rld))), ss(iz)), bs(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3291
    rl_inc = FADD(rl_inc, FDIV(FDIV(FMUL(FMUL(rld, ss(iz)), bs(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3292
    rs_abs = FADD(rs_abs, FDIV(FDIV(FMUL(FMUL(FMUL((FSUB(1.f, albr_u)), rs), ss(iz)), bs(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3293
    gfl = FADD(FSUB(FADD(FMUL((FSUB(1.f, albr_u)), rs), FMUL(emr_u, rld)), FMUL(FMUL(emr_u, sigma), (gfk_pow(tr(id, iz, nwr_u), 4.f)))), sfr(id, iz));
    // module_sf_bep.F:3294
    grdflx_urb = FSUB(grdflx_urb, FDIV(FDIV(FMUL(FMUL(gfl, ss(iz)), bs(id)), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3295
    }
    // module_sf_bep.F:3297
    for (iz = 1; iz <= nzu; iz += 1) {
    // module_sf_bep.F:3298
    rl_emit = FSUB(rl_emit, FDIV(FDIV(FMUL(FMUL((FADD(FMUL(FMUL(emw_u, sigma), (FADD(gfk_pow(tw(((2 * id) - 1), iz, nwr_u), 4.f), gfk_pow(tw((2 * id), iz, nwr_u), 4.f)))), FMUL((FSUB(float(1), emw_u)), (FADD(rlw(((2 * id) - 1), iz), rlw((2 * id), iz)))))), dz_u), pb((iz + 1))), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3300
    rl_inc = FADD(rl_inc, FDIV(FDIV(FMUL(FMUL(((FADD(rlw(((2 * id) - 1), iz), rlw((2 * id), iz)))), dz_u), pb((iz + 1))), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3301
    rs_abs = FADD(rs_abs, FDIV(FDIV(FMUL(FMUL((FMUL((FSUB(1.f, albw_u)), (FADD(rsw(((2 * id) - 1), iz), rsw((2 * id), iz))))), dz_u), pb((iz + 1))), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3302
    gfl = FADD(FSUB(FADD(FMUL((FSUB(1.f, albw_u)), (FADD(rsw(((2 * id) - 1), iz), rsw((2 * id), iz)))), FMUL(emw_u, (FADD(rlw(((2 * id) - 1), iz), rlw((2 * id), iz))))), FMUL(FMUL(emw_u, sigma), (FADD(gfk_pow(tw(((2 * id) - 1), iz, nwr_u), 4.f), gfk_pow(tw((2 * id), iz, nwr_u), 4.f))))), (FADD(sfw(((2 * id) - 1), iz), sfw((2 * id), iz))));
    // module_sf_bep.F:3304
    grdflx_urb = FSUB(grdflx_urb, FDIV(FDIV(FMUL(FMUL(gfl, dz_u), pb((iz + 1))), (FADD(ws(id), bs(id)))), float(ndu)));
    // module_sf_bep.F:3305
    }
    // module_sf_bep.F:3307
    }
    // module_sf_bep.F:3308
    emiss = FDIV((FADD(FADD(emg_u, emw_u), emr_u)), 3.f);
    // module_sf_bep.F:3309
    rl_up = FSUB((FADD(rl_inc, rl_emit)), rld);
    // module_sf_bep.F:3312
    return;
}

// module_sf_bep.F:3321-3377
__device__ void bep_icbep_xy(BepWS &arena, int &error, int iurb, BepArray<float,4> fww_u, BepArray<float,3> fwg_u, BepArray<float,3> fgw_u, BepArray<float,3> fsw_u, BepArray<float,3> fws_u, BepArray<float,2> fsg_u, int ndu, BepArray<float,1> strd, BepArray<float,1> ws, int nzu, BepArray<float,1> z_u) {
    BepFrame frame(arena);
    int id;
    // module_sf_bep.F:3363
    for (int wi=0; wi<fww_u.size(); ++wi) fww_u[wi] = 0.f;
    // module_sf_bep.F:3364
    for (int wi=0; wi<fwg_u.size(); ++wi) fwg_u[wi] = 0.f;
    // module_sf_bep.F:3365
    for (int wi=0; wi<fgw_u.size(); ++wi) fgw_u[wi] = 0.f;
    // module_sf_bep.F:3366
    for (int wi=0; wi<fsw_u.size(); ++wi) fsw_u[wi] = 0.f;
    // module_sf_bep.F:3367
    for (int wi=0; wi<fws_u.size(); ++wi) fws_u[wi] = 0.f;
    // module_sf_bep.F:3368
    for (int wi=0; wi<fsg_u.size(); ++wi) fsg_u[wi] = 0.f;
    // module_sf_bep.F:3370
    for (id = 1; id <= ndu; id += 1) {
    // module_sf_bep.F:3372
    bep_view_factors(arena, error, iurb, nzu, id, strd(id), z_u, ws(id), fww_u, fwg_u, fgw_u, fsg_u, fsw_u, fws_u); if (error) return;
    // module_sf_bep.F:3375
    }
    // module_sf_bep.F:3376
    return;
}

// module_sf_bep.F:3381-3497
__device__ void bep_icbephi_xy(BepWS &arena, int &error, BepArray<float,1> hb_u, BepArray<float,1> hi_urb1d, BepArray<float,1> ss_u, BepArray<float,1> pb_u, int &nzu, BepArray<float,1> z_u) {
    BepFrame frame(arena);
    int iz_u;
    int id;
    int ilu;
    float dtot;
    float hbmax;
    // module_sf_bep.F:3419
    nzu = 0;
    // module_sf_bep.F:3420
    for (int wi=0; wi<ss_u.size(); ++wi) ss_u[wi] = 0.f;
    // module_sf_bep.F:3421
    for (int wi=0; wi<pb_u.size(); ++wi) pb_u[wi] = 0.f;
    // module_sf_bep.F:3425
    dtot = 0.f;
    // module_sf_bep.F:3426
    for (int wi=0; wi<hb_u.size(); ++wi) hb_u[wi] = 0.f;
    // module_sf_bep.F:3428
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3429
    dtot = FADD(dtot, hi_urb1d(ilu));
    // module_sf_bep.F:3430
    }
    // module_sf_bep.F:3432
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3433
    if ((hi_urb1d(ilu) < 0.f)) {
    // module_sf_bep.F:3435
    goto L20;
    // module_sf_bep.F:3436
    }
    // module_sf_bep.F:3437
    }
    // module_sf_bep.F:3439
    if ((dtot > 0.f)) {
    // module_sf_bep.F:3440
    ;
    // module_sf_bep.F:3441
    } else {
    // module_sf_bep.F:3443
    goto L20;
    // module_sf_bep.F:3444
    }
    // module_sf_bep.F:3446
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3447
    hi_urb1d(ilu) = FDIV(hi_urb1d(ilu), dtot);
    // module_sf_bep.F:3448
    }
    // module_sf_bep.F:3450
    hb_u(1) = dz_u;
    // module_sf_bep.F:3451
    for (ilu = 2; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3452
    hb_u(ilu) = FADD(dz_u, hb_u((ilu - 1)));
    // module_sf_bep.F:3453
    }
    // module_sf_bep.F:3459
    hbmax = 0.f;
    // module_sf_bep.F:3461
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3462
    if (((hi_urb1d(ilu) > 0) && (hi_urb1d(ilu) <= 1.f))) {
    // module_sf_bep.F:3463
    hbmax = hb_u(ilu);
    // module_sf_bep.F:3464
    }
    // module_sf_bep.F:3465
    }
    // module_sf_bep.F:3467
    for (iz_u = 1; iz_u <= (nz_um - 1); iz_u += 1) {
    // module_sf_bep.F:3468
    if ((z_u((iz_u + 1)) > hbmax)) { goto L10; }
    // module_sf_bep.F:3469
    }
    // module_sf_bep.F:3471
    L10: ;
    // module_sf_bep.F:3473
    nzu = (iz_u + 1);
    // module_sf_bep.F:3475
    if ((((nzu + 1)) > nz_um)) {
    // module_sf_bep.F:3476
    ; // diagnostic omitted
    // module_sf_bep.F:3477
    { error = 3477; return; }
    // module_sf_bep.F:3478
    }
    // module_sf_bep.F:3480
    for (iz_u = 1; iz_u <= nzu; iz_u += 1) {
    // module_sf_bep.F:3481
    ss_u(iz_u) = 0.f;
    // module_sf_bep.F:3482
    for (ilu = 1; ilu <= nz_um; ilu += 1) {
    // module_sf_bep.F:3483
    if (((z_u(iz_u) <= hb_u(ilu)) && (z_u((iz_u + 1)) > hb_u(ilu)))) {
    // module_sf_bep.F:3485
    ss_u(iz_u) = FADD(ss_u(iz_u), hi_urb1d(ilu));
    // module_sf_bep.F:3486
    }
    // module_sf_bep.F:3487
    }
    // module_sf_bep.F:3488
    }
    // module_sf_bep.F:3490
    pb_u(1) = 1.f;
    // module_sf_bep.F:3491
    for (iz_u = 1; iz_u <= nzu; iz_u += 1) {
    // module_sf_bep.F:3492
    pb_u((iz_u + 1)) = fmaxf(0.f, FSUB(pb_u(iz_u), ss_u(iz_u)));
    // module_sf_bep.F:3493
    }
    // module_sf_bep.F:3495
    L20: ;
    // module_sf_bep.F:3496
    return;
}

// module_sf_bep.F:372-386 first-call block, isolated from column scratch.
extern "C" __global__ void urban_bep_class_init(const UrbanBepTable *tbl, UrbanBepClass *out) {
    if (blockIdx.x || threadIdx.x) return;
    BepWS arena{nullptr,0};
    int error=0;
    // Class kernels have no automatic array workspace: all arrays are out fields.
    BepArray<float,1> alag_u(const_cast<float *>(out->alag_u), 1, nurbm, 1);
    BepArray<float,1> alaw_u(const_cast<float *>(out->alaw_u), 1, nurbm, 1);
    BepArray<float,1> alar_u(const_cast<float *>(out->alar_u), 1, nurbm, 1);
    BepArray<float,1> csg_u(const_cast<float *>(out->csg_u), 1, nurbm, 1);
    BepArray<float,1> csw_u(const_cast<float *>(out->csw_u), 1, nurbm, 1);
    BepArray<float,1> csr_u(const_cast<float *>(out->csr_u), 1, nurbm, 1);
    BepArray<float,1> twini_u(const_cast<float *>(out->twini_u), 1, nurbm, 1);
    BepArray<float,1> trini_u(const_cast<float *>(out->trini_u), 1, nurbm, 1);
    BepArray<float,1> tgini_u(const_cast<float *>(out->tgini_u), 1, nurbm, 1);
    BepArray<float,1> albg_u(const_cast<float *>(out->albg_u), 1, nurbm, 1);
    BepArray<float,1> albw_u(const_cast<float *>(out->albw_u), 1, nurbm, 1);
    BepArray<float,1> albr_u(const_cast<float *>(out->albr_u), 1, nurbm, 1);
    BepArray<float,1> emg_u(const_cast<float *>(out->emg_u), 1, nurbm, 1);
    BepArray<float,1> emw_u(const_cast<float *>(out->emw_u), 1, nurbm, 1);
    BepArray<float,1> emr_u(const_cast<float *>(out->emr_u), 1, nurbm, 1);
    BepArray<float,1> z0g_u(const_cast<float *>(out->z0g_u), 1, nurbm, 1);
    BepArray<float,1> z0r_u(const_cast<float *>(out->z0r_u), 1, nurbm, 1);
    BepArray<int,1> nd_u(const_cast<int *>(out->nd_u), 1, nurbm, 1);
    BepArray<float,2> strd_u(const_cast<float *>(out->strd_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> drst_u(const_cast<float *>(out->drst_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> ws_u(const_cast<float *>(out->ws_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> bs_u(const_cast<float *>(out->bs_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> h_b(const_cast<float *>(out->h_b), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> d_b(const_cast<float *>(out->d_b), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> ss_u(const_cast<float *>(out->ss_u), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> pb_u(const_cast<float *>(out->pb_u), 1, nz_um, nurbm, 1, 1);
    BepArray<int,1> nz_u(const_cast<int *>(out->nz_u), 1, nurbm, 1);
    BepArray<float,1> z_u(const_cast<float *>(out->z_u), 1, nz_um, 1);
    bep_init_para(arena,error,*tbl,alag_u, alaw_u, alar_u, csg_u, csw_u, csr_u, twini_u, trini_u, tgini_u, albg_u, albw_u, albr_u, emg_u, emw_u, emr_u, z0g_u, z0r_u, nd_u, strd_u, drst_u, ws_u, bs_u, h_b, d_b);
    if(!error) bep_icbep(arena,error,nd_u, h_b, d_b, ss_u, pb_u, nz_u, z_u);
    out->error=error;
}

struct UrbanBepViews {
    float fww[648];
    float fwg[36];
    float fgw[36];
    float fsw[36];
    float fws[36];
    float fsg[2];
    int error;
};

// Run the unchanged column initialization once for each class. Scratch has
// the same lane stride as the column arena; outputs have contiguous storage.
extern "C" __global__ void urban_bep_view_init(
    const UrbanBepClass *cls, UrbanBepViews *views, float *scratch) {
    if(threadIdx.x) return;
    int iurb=blockIdx.x+1;
    UrbanBepViews &v=views[iurb-1];
    int error=cls->error;
    if(error || cls->nz_u[iurb-1]<1 || cls->nz_u[iurb-1]+1>nz_um) {
        v.error=error ? error : 2736; return;
    }
    BepWS arena{scratch+(size_t)blockIdx.x*8192*32,0};
    BepArray<float,1> strd(const_cast<float *>(cls->strd_u)+ndm*(iurb-1),1,ndm,1);
    BepArray<float,1> ws(const_cast<float *>(cls->ws_u)+ndm*(iurb-1),1,ndm,1);
    BepArray<float,1> z(const_cast<float *>(cls->z_u),1,nz_um,1);
    BepArray<float,4> fww(v.fww,1,nz_um, nz_um, ndm, 1, 1, 1, 1, 1, true);
    BepArray<float,3> fwg(v.fwg,1,nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fgw(v.fgw,1,nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fsw(v.fsw,1,nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fws(v.fws,1,nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,2> fsg(v.fsg,1,ndm, 1, 1, 1, true);
    bep_icbep_xy(arena,error,iurb,fww,fwg,fgw,fsw,fws,fsg,
        cls->nd_u[iurb-1],strd,ws,cls->nz_u[iurb-1],z);
    v.error=error;
}

// module_sf_bep.F:352-580. Column gather/scatter only; each routine above
// executes the original calculation with 1-based Fortran array indices.
extern "C" __global__ void urban_bep_column(
    const float *frc_urb2d, const int *utype_urb2d,
    const float *dz8w, const float *u_phy,const float *v_phy,
    const float *th_phy,const float *rho,const float *p_phy,
    const float *swdown,const float *glw,const float *cosz_urb2d,const float *omg_urb2d,
    float declin_urb,float dt,const float *lp_urb2d,const float *lb_urb2d,
    const float *hgt_urb2d,const float *hi_urb2d,const UrbanBepClass *cls,
    float *trb_urb4d,float *tw1_urb4d,float *tw2_urb4d,float *tgb_urb4d,
    float *sfw1_urb3d,float *sfw2_urb3d,float *sfr_urb3d,float *sfg_urb3d,
    float *a_u,float *a_v,float *a_t,float *a_e,float *b_u,float *b_v,
    float *b_t,float *b_e,float *b_q,float *dlg,float *dl_u,float *vl,float *sf,
    float *rl_up,float *rs_abs,float *emiss,float *grdflx_urb,int *error_flags,
    float *workspace,int workspace_slots,int nz,int ncol,int num_urban_hi,
    int column_offset,int tile_count,const int *col_index,int use_index,
    const UrbanBepViews *views,int use_view_cache) {
    // Tiles run over col_index (the FRC_URB2D > 0 columns, gathered once by
    // the launcher) when use_index is set, so a sparse city does not spend a
    // tile of workspace on columns that return at once; WRF's own test
    // (module_sf_bep.F:392, FRC_URB2D > 0) still decides every column.
    int lane=blockIdx.x*32+threadIdx.x;
    if(lane>=tile_count) return;
    int col=use_index ? col_index[column_offset+lane] : column_offset+lane;
    if(col>=ncol || !(frc_urb2d[col]>0.f)) return;
    int error=cls->error;
    error_flags[col]=error;
    if(error) return;
    if(num_urban_hi>=nz_um || num_urban_hi<0) {error_flags[col]=348;return;}
    int iurb=utype_urb2d[col];
    if(iurb<1 || iurb>nurbmax) {error_flags[col]=392;return;}
    // module_sf_bep.F:589-905 scratch bound, enforced for direct callers too.
    if(nz<1 || workspace_slots<8192+24*(nz+1)) {error_flags[col]=589;return;}
    BepWS arena{workspace+(size_t)blockIdx.x*workspace_slots*32+threadIdx.x,0};
    BepArray<float,1> alag_u(const_cast<float *>(cls->alag_u), 1, nurbm, 1);
    BepArray<float,1> alaw_u(const_cast<float *>(cls->alaw_u), 1, nurbm, 1);
    BepArray<float,1> alar_u(const_cast<float *>(cls->alar_u), 1, nurbm, 1);
    BepArray<float,1> csg_u(const_cast<float *>(cls->csg_u), 1, nurbm, 1);
    BepArray<float,1> csw_u(const_cast<float *>(cls->csw_u), 1, nurbm, 1);
    BepArray<float,1> csr_u(const_cast<float *>(cls->csr_u), 1, nurbm, 1);
    BepArray<float,1> twini_u(const_cast<float *>(cls->twini_u), 1, nurbm, 1);
    BepArray<float,1> trini_u(const_cast<float *>(cls->trini_u), 1, nurbm, 1);
    BepArray<float,1> tgini_u(const_cast<float *>(cls->tgini_u), 1, nurbm, 1);
    BepArray<float,1> albg_u(const_cast<float *>(cls->albg_u), 1, nurbm, 1);
    BepArray<float,1> albw_u(const_cast<float *>(cls->albw_u), 1, nurbm, 1);
    BepArray<float,1> albr_u(const_cast<float *>(cls->albr_u), 1, nurbm, 1);
    BepArray<float,1> emg_u(const_cast<float *>(cls->emg_u), 1, nurbm, 1);
    BepArray<float,1> emw_u(const_cast<float *>(cls->emw_u), 1, nurbm, 1);
    BepArray<float,1> emr_u(const_cast<float *>(cls->emr_u), 1, nurbm, 1);
    BepArray<float,1> z0g_u(const_cast<float *>(cls->z0g_u), 1, nurbm, 1);
    BepArray<float,1> z0r_u(const_cast<float *>(cls->z0r_u), 1, nurbm, 1);
    BepArray<int,1> nd_u(const_cast<int *>(cls->nd_u), 1, nurbm, 1);
    BepArray<float,2> strd_u(const_cast<float *>(cls->strd_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> drst_u(const_cast<float *>(cls->drst_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> ws_u(const_cast<float *>(cls->ws_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> bs_u(const_cast<float *>(cls->bs_u), 1, ndm, nurbm, 1, 1);
    BepArray<float,2> h_b(const_cast<float *>(cls->h_b), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> d_b(const_cast<float *>(cls->d_b), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> ss_u(const_cast<float *>(cls->ss_u), 1, nz_um, nurbm, 1, 1);
    BepArray<float,2> pb_u(const_cast<float *>(cls->pb_u), 1, nz_um, nurbm, 1, 1);
    BepArray<int,1> nz_u(const_cast<int *>(cls->nz_u), 1, nurbm, 1);
    BepArray<float,1> z_u(const_cast<float *>(cls->z_u), 1, nz_um, 1);
    BepArray<float,1> hb_u(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> hi_urb1d(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> ss_urb(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> pb_urb(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> csg(arena.alloc<float>((ng_u)), 32, ng_u, 1);
    BepArray<float,1> csr(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> csw(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> alag(arena.alloc<float>((ng_u)), 32, ng_u, 1);
    BepArray<float,1> alaw(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,1> alar(arena.alloc<float>((nwr_u)), 32, nwr_u, 1);
    BepArray<float,2> z0(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,1> bs(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,1> ws(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,1> drst(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,1> strd(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,1> ss(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,1> pb(arena.alloc<float>((nz_um)), 32, nz_um, 1);
    BepArray<float,3> tw1d(arena.alloc<float>(((2 * ndm)) * (nz_um) * (nwr_u)), 32, (2 * ndm), nz_um, nwr_u, 1, 1, 1);
    BepArray<float,2> tg1d(arena.alloc<float>((ndm) * (ng_u)), 32, ndm, ng_u, 1, 1);
    BepArray<float,3> tr1d(arena.alloc<float>((ndm) * (nz_um) * (nwr_u)), 32, ndm, nz_um, nwr_u, 1, 1, 1);
    BepArray<float,2> sfw1d(arena.alloc<float>(((2 * ndm)) * (nz_um)), 32, (2 * ndm), nz_um, 1, 1);
    BepArray<float,1> sfg1d(arena.alloc<float>((ndm)), 32, ndm, 1);
    BepArray<float,2> sfr1d(arena.alloc<float>((ndm) * (nz_um)), 32, ndm, nz_um, 1, 1);
    BepArray<float,4> fww_u(arena.alloc<float>((nz_um) * (nz_um) * (ndm) * (1)), 32, nz_um, nz_um, ndm, 1, 1, 1, 1, 1, true);
    BepArray<float,3> fwg_u(arena.alloc<float>((nz_um) * (ndm) * (1)), 32, nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fgw_u(arena.alloc<float>((nz_um) * (ndm) * (1)), 32, nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fsw_u(arena.alloc<float>((nz_um) * (ndm) * (1)), 32, nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,3> fws_u(arena.alloc<float>((nz_um) * (ndm) * (1)), 32, nz_um, ndm, 1, 1, 1, 1, true);
    BepArray<float,2> fsg_u(arena.alloc<float>((ndm) * (1)), 32, ndm, 1, 1, 1, true);
    BepArray<float,1> z1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> ua1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> va1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> pt1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> da1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> pr1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> pt01d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> sf1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> vl1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> a_u1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> a_v1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> a_t1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> a_e1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> b_u1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> b_v1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> b_t1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> b_e1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> dlg1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    BepArray<float,1> dl_u1d(arena.alloc<float>(((nz + 1))), 32, (nz + 1), 1);
    // module_sf_bep.F:352-369, z(kts)=0, mass-level dz accumulated in order.
    z1d(1)=0.f;
    for(int k=1;k<=nz;++k) {
        int q=(k-1)*ncol+col;
        z1d(k+1)=FADD(z1d(k),dz8w[q]);
        ua1d(k)=u_phy[q]; va1d(k)=v_phy[q]; pt1d(k)=th_phy[q];
        da1d(k)=rho[q]; pr1d(k)=p_phy[q]; pt01d(k)=300.f;
        a_u1d(k)=a_v1d(k)=a_t1d(k)=a_e1d(k)=0.f;
        b_u1d(k)=b_v1d(k)=b_t1d(k)=b_e1d(k)=0.f;
    }
    for(int k=1;k<=nz_um;++k) hi_urb1d(k)=k<=num_urban_hi?hi_urb2d[(k-1)*ncol+col]:0.f;
    int nzurb=0,nzurban=0;
    bep_icbephi_xy(arena,error,hb_u,hi_urb1d,ss_urb,pb_urb,nzurb,z_u);
    if(error){error_flags[col]=error;return;}
    bep_param(arena,error,iurb,nz_u(iurb),nzurb,nzurban,nd_u(iurb),
        csg_u,csg,alag_u,alag,csr_u,csr,alar_u,alar,csw_u,csw,alaw_u,alaw,
        ws_u,ws,bs_u,bs,z0g_u,z0r_u,z0,strd_u,strd,drst_u,drst,
        ss_u,ss_urb,ss,pb_u,pb_urb,pb,lp_urb2d[col],lb_urb2d[col],hgt_urb2d[col],frc_urb2d[col]);
    // module_sf_bep.F:2736-2758 assumes one spare urban interface.
    if(nzurban<1 || nzurban+1>nz_um) {error_flags[col]=2736;return;}
    // z_u and nd_u are shared with initialization. Compare every remaining
    // effective input by its float32 word, including signed zero.
    bool cached=use_view_cache && !views[iurb-1].error && nzurban==nz_u(iurb);
    for(int id=1;id<=nd_u(iurb) && cached;++id)
        cached=__float_as_uint(ws(id))==__float_as_uint(ws_u(id,iurb)) &&
               __float_as_uint(strd(id))==__float_as_uint(strd_u(id,iurb));
    if(cached) {
        const UrbanBepViews &v=views[iurb-1];
        for(int k=0;k<648;++k) fww_u[k]=v.fww[k];
        for(int k=0;k<36;++k) fwg_u[k]=v.fwg[k];
        for(int k=0;k<36;++k) fgw_u[k]=v.fgw[k];
        for(int k=0;k<36;++k) fsw_u[k]=v.fsw[k];
        for(int k=0;k<36;++k) fws_u[k]=v.fws[k];
        for(int k=0;k<2;++k) fsg_u[k]=v.fsg[k];
    } else {
    bep_icbep_xy(arena,error,iurb,fww_u,fwg_u,fgw_u,fsw_u,fws_u,fsg_u,
        nd_u(iurb),strd,ws,nzurban,z_u);
    }
    // module_sf_bep.F:318-346, collapsed maps. iii-1 is exactly the
    // external zero-based layer, not the Fortran scratch storage order.
    for(int id=1;id<=ndm;++id) {
        for(int iz=1;iz<=nz_um;++iz) {
            int zd=(iz-1)*ndm+id-1;
            sfw1d(2*id-1,iz)=sfw1_urb3d[zd*ncol+col];
            sfw1d(2*id,iz)=sfw2_urb3d[zd*ncol+col];
            sfr1d(id,iz)=sfr_urb3d[zd*ncol+col];
            for(int iw=1;iw<=nwr_u;++iw) {
                int zwd=((iz-1)*nwr_u+iw-1)*ndm+id-1;
                tw1d(2*id-1,iz,iw)=tw1_urb4d[zwd*ncol+col];
                tw1d(2*id,iz,iw)=tw2_urb4d[zwd*ncol+col];
                tr1d(id,iz,iw)=trb_urb4d[zwd*ncol+col];
            }
        }
        for(int ig=1;ig<=ng_u;++ig) tg1d(id,ig)=tgb_urb4d[((ig-1)*ndm+id-1)*ncol+col];
        sfg1d(id)=sfg_urb3d[(id-1)*ncol+col];
    }
    // module_sf_bep.F:479-484, time_h and angle do not feed the live call.
    float zr=glibc_acosf(cosz_urb2d[col]);
    bep_bep1d(arena,error,iurb,1,nz+1,1,nz,z1d,dt,ua1d,va1d,pt1d,da1d,pr1d,pt01d,
        zr,declin_urb,omg_urb2d[col],swdown[col],glw[col],alag,alaw,alar,csg,csw,csr,
        albg_u(iurb),albw_u(iurb),albr_u(iurb),emg_u(iurb),emw_u(iurb),emr_u(iurb),
        fww_u,fwg_u,fgw_u,fsw_u,fws_u,fsg_u,z0,nd_u(iurb),strd,drst,ws,bs,ss,pb,
        nzurban,z_u,tw1d,tg1d,tr1d,sfw1d,sfg1d,sfr1d,
        a_u1d,a_v1d,a_t1d,a_e1d,b_u1d,b_v1d,b_t1d,b_e1d,dlg1d,dl_u1d,sf1d,vl1d,
        rl_up[col],rs_abs[col],emiss[col],grdflx_urb[col]);
    error_flags[col]=error;
    if(error) return;
    for(int id=1;id<=ndm;++id) {
        for(int iz=1;iz<=nz_um;++iz) {
            int zd=(iz-1)*ndm+id-1;
            sfw1_urb3d[zd*ncol+col]=sfw1d(2*id-1,iz);
            sfw2_urb3d[zd*ncol+col]=sfw1d(2*id,iz);
            sfr_urb3d[zd*ncol+col]=sfr1d(id,iz);
            for(int iw=1;iw<=nwr_u;++iw) {
                int zwd=((iz-1)*nwr_u+iw-1)*ndm+id-1;
                tw1_urb4d[zwd*ncol+col]=tw1d(2*id-1,iz,iw);
                tw2_urb4d[zwd*ncol+col]=tw1d(2*id,iz,iw);
                trb_urb4d[zwd*ncol+col]=tr1d(id,iz,iw);
            }
        }
        for(int ig=1;ig<=ng_u;++ig) tgb_urb4d[((ig-1)*ndm+id-1)*ncol+col]=tg1d(id,ig);
        sfg_urb3d[(id-1)*ncol+col]=sfg1d(id);
    }
    for(int k=1;k<=nz;++k) {
        int q=(k-1)*ncol+col;
        a_u[q]=a_u1d(k);a_v[q]=a_v1d(k);a_t[q]=a_t1d(k);a_e[q]=a_e1d(k);
        b_u[q]=b_u1d(k);b_v[q]=b_v1d(k);b_t[q]=b_t1d(k);b_e[q]=b_e1d(k);
        b_q[q]=0.f;dlg[q]=dlg1d(k);dl_u[q]=dl_u1d(k);vl[q]=vl1d(k);sf[q]=sf1d(k);
    }
    sf[nz*ncol+col]=sf1d(nz+1);
}
