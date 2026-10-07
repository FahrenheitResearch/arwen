// Freitas variants: WRF f52c197e chem/module_chem_plumerise_scalar.F and
// GSL 3e6660c6 physics/smoke_dust/module_smoke_plumerise.F90.
// All arithmetic uses explicit rounding. Local state is zeroed per column;
// state is retained between imm passes and landuse groups of that column.
// The state-poison and order experiments are recorded in fixture PROVENANCE.
// WRF INITIAL:979-1000 writes only 1..kmt. scl_advectc:1294 reads rho(kmt+1)
// in an unused top flux; set_flam_vert:377-403 reads retained W_VMD tails for
// its unused VMD diagnostic. WBAR/DQSDZ are read by initial WATERBAL but have
// no measured output effect in the supplied cases. Same-column carry remains.
// WRF EVAPORATE:1998/SUBLIMATE:2268 can read CVI(1), which fallpart:1688-1707
// never writes. A -1e10 poison changes one case's injection; zero is defined.
// Forward/reverse retained columns change profile tails, not injection/top.
// -finit-real=snan/-finit-integer=-999999 changes no published solver words.
// GSL dbg_opt is false in the oracle; module_plumerise.F90:80-93's unwritten
// icall is a print gate only. No debug I/O is part of this transcription.
// Public sources:
// https://github.com/wrf-model/WRF/tree/f52c197ed39d12e087d02c50f412d90d418f6186/chem
// https://github.com/ufs-community/ccpp-physics/tree/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust
// WRF source is public domain; GSL-derived statements are Apache-2.0.
// Generated source-line transcription: tools/chem_wrf471_oracle/transcribe_plume.py.
#define PL_WS_ROWS 131  // workspace rows per fire column (see freitas_columns)
__device__ inline float pl_fadd(float a,float b) { return FADD(a,b); }
__device__ inline float pl_fsub(float a,float b) { return FSUB(a,b); }
__device__ inline float pl_fmul(float a,float b) { return FMUL(a,b); }
__device__ inline float pl_fdiv(float a,float b) { return FDIV(a,b); }
__device__ inline float pl_fmin(float a,float b) { return fminf(a,b); }
__device__ inline float pl_fmax(float a,float b) { return fmaxf(a,b); }
__device__ inline float pl_fabs(float x) { return fabsf(x); }
__device__ inline float pl_fsqrt(float x) { return FSQRT(x); }
__device__ inline float pl_fexp(float x) { return gfk_exp(x); }
__device__ inline float pl_fpow(float x,float y) {
    if(x<0.0f && isfinite(x) && isfinite(y) && y!=truncf(y)) return __uint_as_float(0xffc00000u);
    return gfk_pow(x,y);
}
__device__ inline float pl_fpowi(float x,int n) {
    unsigned int p=n<0?-n:n;
    float result=(p&1)?x:1.0f;
    while ((p>>=1)) { x=FMUL(x,x); if(p&1) result=FMUL(result,x); }
    return n<0?FDIV(1.0f,result):result;
}
__device__ inline float pl_fsum(float* p,int lo,int hi) {
    float result=0.0f; for(int k=lo;k<=hi;++k) result=FADD(result,p[k]); return result;
}
__device__ void wrf_get_env_condition(float s[][202], int k1, int k2, int& kmt, int wind_eff) ;
__device__ void wrf_set_grid(float s[][202]) ;
__device__ void wrf_set_flam_vert(float s[][202], float* ztopmax, int& k1, int& k2, int nkp, float* zzcon, float (*w_vmd)[202], float (*vmd)[202]) ;
__device__ void wrf_get_fire_properties(float s[][202], int imm, int iveg_ag, float burnt_area, float std_burnt_area, float (*heat_flux)[202]) ;
__device__ void wrf_makeplume(float s[][202], int kmt, float& ztopmax, int ixx, int imm) ;
__device__ void wrf_burn(float s[][202], float& eflux, float& water) ;
__device__ void wrf_lbound(float s[][202]) ;
__device__ void wrf_initial(float s[][202], int kmt) ;
__device__ void wrf_damp_grav_wave(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* w, float* t, float* tt, float* qv, float* qh, float* qi, float* qc, float* te, float* pe, float* qvenv) ;
__device__ void wrf_friction(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* var1, float* vart, float* var2) ;
__device__ void wrf_vel_advectc_plumerise(float s[][202], int m1, float* wc, float* wt, float* rho, float* dzm) ;
__device__ void wrf_hadvance_plumerise(float s[][202], int iac, int m1, float dt, float* wc, float* wt, float* wp, int mintime) ;
__device__ void wrf_predict_plumerise(float s[][202], int npts, float* ac, float* ap, float* fa, float* af, int iac, float dtlp, float epsu) ;
__device__ void wrf_buoyancy_plumerise(float s[][202], int m1, float* t, float* te, float* qv, float* qvenv, float* qh, float* qi, float* qc, float* wt, float* scr1) ;
__device__ void wrf_entrainment(float s[][202], int m1, float* w, float* wt, float* radius, float alpha) ;
__device__ void wrf_scl_advectc_plumerise(float s[][202], char varn, int mzp) ;
__device__ void wrf_fa_zc_plumerise(float s[][202], int m1, float* scp, float* scr1, float* vt3dc, float* vt3df, float* vt3dg, float* vt3dk, float* vctr1, float* vctr2) ;
__device__ void wrf_advtndc_plumerise(float s[][202], int m1, float* scp, float* sca, float* sct, float dtl) ;
__device__ void wrf_tend0_plumerise(float s[][202]) ;
__device__ void wrf_scl_misc(float s[][202], int m1) ;
__device__ void wrf_scl_dyn_entrain(float s[][202], int m1, int nkp, float wbar, float* w, float adiabat, float alpha, float* radius, float* tt, float* t, float* te, float* qvt, float* qv, float* qvenv, float* qct, float* qc, float* qht, float* qh, float* qit, float* qi, float* vel_e, float* vel_p, float* vel_t, float* rad_p, float* rad_t) ;
__device__ void wrf_visc_w(float s[][202], int m1, int deltak, int kmt) ;
__device__ void wrf_update_plumerise(float s[][202], int m1, char varn) ;
__device__ void wrf_fallpart(float s[][202], int m1) ;
__device__ void wrf_waterbal(float s[][202]) ;
__device__ void wrf_evaporate(float s[][202]) ;
__device__ void wrf_convert(float s[][202]) ;
__device__ void wrf_convert2(float s[][202]) ;
__device__ void wrf_sublimate(float s[][202]) ;
__device__ void wrf_glaciate(float s[][202]) ;
__device__ void wrf_melt(float s[][202]) ;
__device__ void wrf_htint(float s[][202], int nzz1, float* vctra, float* eleva, int nzz2, float* vctrb, float* elevb) ;
__device__ float wrf_esat_pr(float s[][202], float tem) ;
__device__ void gsl_get_env_condition(float s[][202], int k1, int k2, int& kmt, int wind_eff, float g, float cp, float rgas, float cpor) ;
__device__ void gsl_set_grid(float s[][202]) ;
__device__ void gsl_set_flam_vert(float s[][202], float* ztopmax, int& k1, int& k2, int nkp, float* zzcon) ;
__device__ void gsl_get_fire_properties(float s[][202], int imm, float burnt_area, float frp) ;
__device__ void gsl_makeplume(float s[][202], int kmt, float& ztopmax, int ixx, int imm, int mpiid, float alpha) ;
__device__ void gsl_burn(float s[][202], float& eflux, float& water) ;
__device__ void gsl_lbound(float s[][202], float alpha) ;
__device__ void gsl_initial(float s[][202], int kmt, float alpha) ;
__device__ void gsl_damp_grav_wave(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* w, float* t, float* tt, float* qv, float* qh, float* qi, float* qc, float* te, float* pe, float* qvenv) ;
__device__ void gsl_friction(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* var1, float* vart, float* var2) ;
__device__ void gsl_vel_advectc_plumerise(float s[][202], int m1, float* wc, float* wt, float* rho, float* dzm) ;
__device__ void gsl_hadvance_plumerise(float s[][202], int iac, int m1, float dt, float* wc, float* wt, float* wp, int mintime) ;
__device__ void gsl_predict_plumerise(float s[][202], int npts, float* ac, float* ap, float* fa, float* af, int iac, float dtlp, float epsu) ;
__device__ void gsl_buoyancy_plumerise(float s[][202], int m1, float* t, float* te, float* qv, float* qvenv, float* qh, float* qi, float* qc, float* wt, float* scr1) ;
__device__ void gsl_entrainment(float s[][202], int m1, float* w, float* wt, float* radius, float alpha) ;
__device__ void gsl_scl_advectc_plumerise(float s[][202], char varn, int mzp) ;
__device__ void gsl_fa_zc_plumerise(float s[][202], int m1, float* scp, float* scr1, float* vt3dc, float* vt3df, float* vt3dg, float* vt3dk, float* vctr1, float* vctr2) ;
__device__ void gsl_advtndc_plumerise(float s[][202], int m1, float* scp, float* sca, float* sct, float dtl) ;
__device__ void gsl_tend0_plumerise(float s[][202]) ;
__device__ void gsl_scl_misc(float s[][202], int m1, float alpha) ;
__device__ void gsl_scl_dyn_entrain(float s[][202], int m1, int nkp, float wbar, float* w, float adiabat, float alpha, float* radius, float* tt, float* t, float* te, float* qvt, float* qv, float* qvenv, float* qct, float* qc, float* qht, float* qh, float* qit, float* qi, float* vel_e, float* vel_p, float* vel_t, float* rad_p, float* rad_t) ;
__device__ void gsl_visc_w(float s[][202], int m1, int deltak, int kmt) ;
__device__ void gsl_update_plumerise(float s[][202], int m1, char varn) ;
__device__ void gsl_fallpart(float s[][202], int m1) ;
__device__ void gsl_waterbal(float s[][202]) ;
__device__ void gsl_evaporate(float s[][202]) ;
__device__ void gsl_convert(float s[][202]) ;
__device__ void gsl_sublimate(float s[][202]) ;
__device__ void gsl_glaciate(float s[][202]) ;
__device__ void gsl_melt(float s[][202]) ;
__device__ void gsl_htint(float s[][202], int nzz1, float* vctra, float* eleva, int nzz2, float* vctrb, float* elevb) ;
__device__ float gsl_esat_pr(float s[][202], float tem) ;
// chem/module_chem_plumerise_scalar.F:212-287.
__device__ void wrf_get_env_condition(float s[][202], int k1, int k2, int& kmt, int wind_eff) {
    int k = 0;
    int kcon = 0;
    int klcl = 0;
    int nk = 0;
    int nkmid = 0;
    int i = 0;
    float znz = 0;
    float themax = 0;
    float tlll = 0;
    float plll = 0;
    float rlll = 0;
    float zlll = 0;
    float dzdd = 0;
    float dzlll = 0;
    float tlcl = 0;
    float plcl = 0;
    float dzlcl = 0;
    float dummy = 0;
    int n_setgrid = 0;
    int _a = 0;
    if ((n_setgrid == 0)) {
        n_setgrid = 1;
        wrf_set_grid(s);
    }
    znz = s[53][k2];
    for (k=200; k>=1; k+=-1) {
        if ((s[26][k] < znz)) {
            break;
        }
    }
    kmt = min(k, 199);
    nk = ((k2 - k1) + 1);
    wrf_htint(s, nk, s[44], s[53], kmt, s[100], s[26]);
    wrf_htint(s, nk, s[45], s[53], kmt, s[101], s[26]);
    wrf_htint(s, nk, s[47], s[53], kmt, s[35], s[26]);
    wrf_htint(s, nk, s[48], s[53], kmt, s[40], s[26]);
    for (k=1; k<=kmt; k+=1) {
        s[40][k] = pl_fmax(s[40][k], __uint_as_float(0x322bcc77u));
    }
    s[34][1] = s[49][1];
    for (k=1; k<=kmt; k+=1) {
        s[36][k] = pl_fmul(s[35][k], pl_fadd(__uint_as_float(0x3f800000u), pl_fmul(__uint_as_float(0x3f1c28f6u), s[40][k])));
    }
    for (k=2; k<=kmt; k+=1) {
        s[34][k] = pl_fsub(s[34][(k - 1)], pl_fdiv(pl_fmul(__uint_as_float(0x419cf5c3u), pl_fsub(s[26][k], s[26][(k - 1)])), pl_fadd(s[36][k], s[36][(k - 1)])));
    }
    for (k=1; k<=kmt; k+=1) {
        s[39][k] = pl_fdiv(pl_fmul(s[35][k], s[34][k]), __uint_as_float(0x447b2000u));
        s[38][k] = pl_fmul(pl_fpow(pl_fdiv(s[34][k], __uint_as_float(0x447b2000u)), __uint_as_float(0x405fffffu)), __uint_as_float(0x47c35000u));
        s[42][k] = pl_fdiv(s[38][k], pl_fmul(pl_fmul(__uint_as_float(0x438f8000u), s[39][k]), pl_fadd(__uint_as_float(0x3f800000u), pl_fmul(__uint_as_float(0x3f1c28f6u), s[40][k]))));
        s[102][k] = pl_fsqrt(pl_fadd(pl_fpowi(s[100][k], 2), pl_fpowi(s[101][k], 2)));
    }
    if ((wind_eff < 1)) {
        for (int _a=1; _a<=kmt; ++_a) s[102][_a] = __uint_as_float(0x00000000u);
    }
    for (k=1; k<=kmt; k+=1) {
        s[38][k] = pl_fmul(s[38][k], __uint_as_float(0x3a83126fu));
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:294-322.
__device__ void wrf_set_grid(float s[][202]) {
    int k = 0;
    int mzp = 0;
    s[56][0] = __uint_as_float(0x42c80000u);
    mzp = 200;
    s[26][1] = s[77][0];
    s[25][1] = s[77][0];
    s[26][2] = pl_fadd(s[26][1], pl_fmul(__uint_as_float(0x3f000000u), s[56][0]));
    s[25][2] = pl_fadd(s[25][1], s[56][0]);
    for (k=3; k<=mzp; k+=1) {
        s[26][k] = pl_fadd(s[26][(k - 1)], s[56][0]);
        s[25][k] = pl_fadd(s[25][(k - 1)], s[56][0]);
    }
    for (k=1; k<=(mzp - 1); k+=1) {
        s[23][k] = pl_fdiv(__uint_as_float(0x3f800000u), pl_fsub(s[26][(k + 1)], s[26][k]));
    }
    s[23][mzp] = s[23][(mzp - 1)];
    for (k=2; k<=mzp; k+=1) {
        s[24][k] = pl_fdiv(__uint_as_float(0x3f800000u), pl_fsub(s[25][k], s[25][(k - 1)]));
    }
    s[24][1] = pl_fdiv(pl_fmul(s[24][2], s[24][2]), s[24][3]);
    return;
}

// chem/module_chem_plumerise_scalar.F:328-410.
__device__ void wrf_set_flam_vert(float s[][202], float* ztopmax, int& k1, int& k2, int nkp, float* zzcon, float (*w_vmd)[202], float (*vmd)[202]) {
    int imm = 0;
    int k = 0;
    int* k_lim = reinterpret_cast<int*>(s[120]);  // workspace row 120: was a 202-word local array
    for (int _z=0; _z<202; ++_z) k_lim[_z] = 0;
    float w_thresold = 0;
    float xxx = 0;
    int k_initial = 0;
    int k_final = 0;
    int ko = 0;
    int kk4 = 0;
    int kl = 0;
    int _a = 0;
    for (imm=1; imm<=2; imm+=1) {
        for (k=1; k<=(nkp - 1); k+=1) {
            if ((zzcon[k] > ztopmax[imm])) {
                break;
            }
        }
        k_lim[imm] = k;
    }
    k1 = max(3, k_lim[1]);
    k2 = max(3, k_lim[2]);
    if ((k2 < k1)) {
        k2 = k1;
    }
    w_thresold = __uint_as_float(0x3f800000u);
    for (imm=1; imm<=2; imm+=1) {
        for (int _a=1; _a<=nkp; ++_a) vmd[imm][_a] = __uint_as_float(0x00000000u);
        xxx = __uint_as_float(0x00000000u);
        k_initial = 0;
        k_final = 0;
        for (ko=(nkp - 10); ko>=2; ko+=-1) {
            if ((w_vmd[imm][ko] < w_thresold)) {
                continue;
            }
            if ((k_final == 0)) {
                k_final = ko;
            }
            if ((pl_fsub(w_vmd[imm][ko], __uint_as_float(0x3f800000u)) > w_vmd[imm][(ko - 1)])) {
                k_initial = ko;
                break;
            }
        }
        if (((k_final > 0) && (k_initial > 0))) {
            k_initial = int(pl_fmul((k_final + k_initial), __uint_as_float(0x3f000000u)));
            kk4 = ((k_final - k_initial) + 2);
            for (ko=1; ko<=(kk4 - 1); ko+=1) {
                kl = ((ko + k_initial) - 1);
                vmd[imm][kl] = pl_fmul(pl_fdiv(pl_fmul(__uint_as_float(0x40c00000u), float(ko)), pl_fpowi(float(kk4), 2)), pl_fsub(__uint_as_float(0x3f800000u), pl_fdiv(float(ko), float(kk4))));
            }
            if ((pl_fsum(vmd[imm], 1, nkp) != __uint_as_float(0x3f800000u))) {
                xxx = pl_fdiv(pl_fsub(__uint_as_float(0x3f800000u), pl_fsum(vmd[imm], 1, nkp)), float(((k_final - k_initial) + 1)));
                for (ko=k_initial; ko<=k_final; ko+=1) {
                    vmd[imm][ko] = pl_fadd(vmd[imm][ko], xxx);
                }
            }
        }
    }
}

// chem/module_chem_plumerise_scalar.F:417-535.
__device__ void wrf_get_fire_properties(float s[][202], int imm, int iveg_ag, float burnt_area, float std_burnt_area, float (*heat_flux)[202]) {
    int moist = 0;
    int i = 0;
    int icount = 0;
    float bfract = 0;
    float effload = 0;
    float heat = 0;
    float hinc = 0;
    float heat_fluxw = 0;
    s[81][0] = burnt_area;
    heat_fluxw = pl_fmul(heat_flux[iveg_ag][imm], __uint_as_float(0x447a0000u));
    s[92][0] = 53;
    s[87][0] = __uint_as_float(0x41200000u);
    moist = 10;
    s[93][0] = (int(s[92][0]) + 2);
    heat = __uint_as_float(0x4b933f50u);
    s[83][0] = __uint_as_float(0x3d4ccccdu);
    s[93][0] = (int(s[93][0]) * 60);
    s[82][0] = pl_fsqrt(pl_fdiv(s[81][0], __uint_as_float(0x40490fd0u)));
    s[86][0] = pl_fdiv(moist, __uint_as_float(0x42c80000u));
    for (i=1; i<=200; i+=1) {
        s[85][i] = __uint_as_float(0x38d1b717u);
    }
    s[90][0] = pl_fmul(int(s[92][0]), __uint_as_float(0x42700000u));
    bfract = __uint_as_float(0x3f800000u);
    effload = pl_fmul(s[87][0], bfract);
    icount = 1;
    if ((int(s[92][0]) > 200)) {
        return;
    }
    while ((icount <= int(s[92][0]))) {
        s[85][icount] = pl_fmul(heat_fluxw, __uint_as_float(0x3f0ccccdu));
        icount = (icount + 1);
    }
    if ((0 != 1)) {
        hinc = pl_fdiv(s[85][1], __uint_as_float(0x40800000u));
        s[85][1] = __uint_as_float(0x3dcccccdu);
        s[85][2] = hinc;
        s[85][3] = pl_fmul(__uint_as_float(0x40000000u), hinc);
        s[85][4] = pl_fmul(__uint_as_float(0x40400000u), hinc);
    } else {
        if ((imm == 1)) {
            hinc = pl_fdiv(s[85][1], __uint_as_float(0x40800000u));
            s[85][1] = __uint_as_float(0x3dcccccdu);
            s[85][2] = hinc;
            s[85][3] = pl_fmul(__uint_as_float(0x40000000u), hinc);
            s[85][4] = pl_fmul(__uint_as_float(0x40400000u), hinc);
        } else {
            hinc = pl_fdiv(pl_fsub(s[85][1], pl_fmul(pl_fmul(heat_flux[iveg_ag][(imm - 1)], __uint_as_float(0x447a0000u)), __uint_as_float(0x3f0ccccdu))), __uint_as_float(0x40800000u));
            s[85][1] = pl_fadd(pl_fmul(pl_fmul(heat_flux[iveg_ag][(imm - 1)], __uint_as_float(0x447a0000u)), __uint_as_float(0x3f0ccccdu)), __uint_as_float(0x3dcccccdu));
            s[85][2] = pl_fadd(s[85][1], hinc);
            s[85][3] = pl_fadd(s[85][2], hinc);
            s[85][4] = pl_fadd(s[85][3], hinc);
        }
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:600-831.
__device__ void wrf_makeplume(float s[][202], int kmt, float& ztopmax, int ixx, int imm) {
    char varn = 0;
    int izprint = 0;
    int iconv = 0;
    int itime = 0;
    int k = 0;
    int kk = 0;
    int kkmax = 0;
    int deltak = 0;
    int ilastprint = 0;
    int nrectotal = 0;
    int i_micro = 0;
    int n_sub_step = 0;
    float wmax = 0;
    float rmaxtime = 0;
    float es = 0;
    float esat = 0;
    float heat = 0;
    float dt_save = 0;
    char cixx = 0;
    float delz_thresold = __uint_as_float(0x42c80000u);
    s[108][0] = 0;
    s[60][0] = __uint_as_float(0x40000000u);
    s[59][0] = __uint_as_float(0x43fa0000u);
    nrectotal = 150;
    s[91][0] = 1;
    ztopmax = __uint_as_float(0x00000000u);
    s[79][0] = __uint_as_float(0x00000000u);
    s[89][0] = __uint_as_float(0x00000000u);
    s[88][0] = __uint_as_float(0x3f800000u);
    wmax = __uint_as_float(0x3f800000u);
    kkmax = 10;
    deltak = 20;
    ilastprint = 0;
    s[63][0] = 1;
    wrf_initial(s, kmt);
    izprint = 0;
    rmaxtime = float(int(s[93][0]));
    while ((s[89][0] <= rmaxtime)) {
        s[62][0] = min(kmt, (kkmax + deltak));
        s[88][0] = pl_fmin(__uint_as_float(0x40a00000u), pl_fdiv(pl_fsub(s[25][2], s[25][1]), pl_fmul(s[60][0], wmax)));
        s[89][0] = pl_fadd(s[89][0], s[88][0]);
        s[108][0] += 1;
        s[91][0] = (1 + (int(s[89][0]) / 60));
        wmax = __uint_as_float(0x3f800000u);
        wrf_tend0_plumerise(s);
        s[63][0] = 1;
        wrf_lbound(s);
        wrf_vel_advectc_plumerise(s, int(s[62][0]), s[15], s[16], s[9], s[23]);
        wrf_scl_advectc_plumerise(s, 's', int(s[62][0]));
        wrf_scl_misc(s, int(s[62][0]));
        wrf_scl_dyn_entrain(s, int(s[62][0]), 200, s[73][0], s[0], s[72][0], s[83][0], s[84], s[17], s[1], s[39], s[18], s[2], s[40], s[19], s[3], s[20], s[4], s[21], s[5], s[102], s[103], s[105], s[104], s[106]);
        wrf_damp_grav_wave(s, 1, int(s[62][0]), deltak, s[88][0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40]);
        dt_save = s[88][0];
        n_sub_step = 3;
        s[88][0] = pl_fdiv(s[88][0], float(n_sub_step));
        for (i_micro=1; i_micro<=n_sub_step; i_micro+=1) {
            wrf_fallpart(s, int(s[62][0]));
            for (s[63][0]=2; int(s[63][0])<=(int(s[62][0]) - 1); s[63][0]+=1) {
                s[73][0] = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][int(s[63][0])], s[0][(int(s[63][0]) - 1)]));
                es = wrf_esat_pr(s, s[1][int(s[63][0])]);
                s[12][int(s[63][0])] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][int(s[63][0])], es));
                s[11][int(s[63][0])] = es;
                s[9][int(s[63][0])] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][int(s[63][0])]), s[1][int(s[63][0])]);
                if ((s[0][int(s[63][0])] >= __uint_as_float(0x00000000u))) {
                    s[57][0] = pl_fdiv(pl_fsub(s[12][(int(s[63][0]) + 1)], s[12][(int(s[63][0]) - 1)]), pl_fsub(s[26][(int(s[63][0]) + 1)], s[26][(int(s[63][0]) - 1)]));
                } else {
                    s[57][0] = pl_fdiv(pl_fsub(s[12][(int(s[63][0]) + 1)], s[12][(int(s[63][0]) - 1)]), pl_fsub(s[26][(int(s[63][0]) + 1)], s[26][(int(s[63][0]) - 1)]));
                }
                wrf_waterbal(s);
            }
        }
        s[88][0] = dt_save;
        wrf_visc_w(s, int(s[62][0]), deltak, kmt);
        wrf_update_plumerise(s, int(s[62][0]), 's');
        wrf_hadvance_plumerise(s, 1, int(s[62][0]), s[88][0], s[15], s[16], s[0], int(s[91][0]));
        wrf_buoyancy_plumerise(s, int(s[62][0]), s[1], s[39], s[2], s[40], s[4], s[5], s[3], s[16], s[33]);
        wrf_entrainment(s, int(s[62][0]), s[0], s[16], s[84], s[83][0]);
        wrf_update_plumerise(s, int(s[62][0]), 'w');
        wrf_hadvance_plumerise(s, 2, int(s[62][0]), s[88][0], s[15], s[16], s[0], int(s[91][0]));
        for (k=2; k<=int(s[62][0]); k+=1) {
            es = wrf_esat_pr(s, s[1][k]);
            s[12][k] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][k], es));
            s[11][k] = es;
            s[10][k] = pl_fsub(s[1][k], s[39][k]);
            s[9][k] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][k]), s[1][k]);
            if ((pl_fabs(s[15][k]) > wmax)) {
                wmax = pl_fabs(s[15][k]);
            }
        }
        wrf_damp_grav_wave(s, 2, int(s[62][0]), deltak, s[88][0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40]);
        for (k=2; k<=int(s[62][0]); k+=1) {
            s[84][k] = s[104][k];
        }
        kk = 1;
        while ((s[0][kk] > __uint_as_float(0x3f800000u))) {
            kk = (kk + 1);
            s[79][0] = s[25][kk];
        }
        s[107][int(s[91][0])] = s[79][0];
        ztopmax = pl_fmax(s[79][0], ztopmax);
        kkmax = max(kk, kkmax);
        if ((int(s[91][0]) > 10)) {
            if ((pl_fabs(pl_fsub(s[107][int(s[91][0])], s[107][(int(s[91][0]) - 10)])) < delz_thresold)) {
                for (k=2; k<=int(s[62][0]); k+=1) {
                    (s+94)[imm][k] = s[0][k];
                }
                break;
            }
        }
    }
    if (imm==1) {
        s[111][0] = ztopmax;
    } else {
        s[112][0] = ztopmax;
    }
    if (imm==1) {
        s[109][0] = s[108][0];
    } else {
        s[110][0] = s[108][0];
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:841-865.
__device__ void wrf_burn(float s[][202], float& eflux, float& water) {
    if ((s[89][0] > s[90][0])) {
        eflux = __uint_as_float(0x358637bdu);
        water = __uint_as_float(0x00000000u);
        return;
    } else {
        eflux = s[85][int(s[91][0])];
        water = pl_fdiv(pl_fmul(pl_fmul(eflux, pl_fdiv(s[88][0], __uint_as_float(0x4b933f50u))), pl_fadd(__uint_as_float(0x3f000000u), s[86][0])), __uint_as_float(0x3f0ccccdu));
        water = pl_fmul(water, __uint_as_float(0x447a0000u));
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:885-964.
__device__ void wrf_lbound(float s[][202]) {
    float es = 0;
    float esat = 0;
    float eflux = 0;
    float water = 0;
    float pres = 0;
    float c1 = 0;
    float c2 = 0;
    float f = 0;
    float zv = 0;
    float denscor = 0;
    float xwater = 0;
    s[4][1] = s[4][2];
    s[5][1] = s[5][2];
    s[3][1] = __uint_as_float(0x00000000u);
    wrf_burn(s, eflux, water);
    pres = pl_fmul(s[38][1], __uint_as_float(0x447a0000u));
    c1 = pl_fdiv(__uint_as_float(0x40a00000u), pl_fmul(__uint_as_float(0x40c00000u), s[83][0]));
    c2 = pl_fmul(__uint_as_float(0x3f666666u), s[83][0]);
    f = pl_fdiv(eflux, pl_fmul(pl_fmul(pres, __uint_as_float(0x447b2666u)), __uint_as_float(0x40490fd0u)));
    f = pl_fmul(pl_fmul(__uint_as_float(0x452ff46eu), f), s[81][0]);
    zv = pl_fmul(c1, s[82][0]);
    s[0][1] = pl_fdiv(pl_fmul(c1, pl_fpow(pl_fmul(c2, f), __uint_as_float(0x3eaaaaabu))), pl_fpow(zv, __uint_as_float(0x3eaaaaabu)));
    denscor = pl_fdiv(pl_fdiv(pl_fdiv(pl_fmul(c1, f), __uint_as_float(0x411ced67u)), pl_fpow(pl_fmul(c2, f), __uint_as_float(0x3eaaaaabu))), pl_fpow(zv, __uint_as_float(0x3fd55555u)));
    s[1][1] = pl_fdiv(s[39][1], pl_fsub(__uint_as_float(0x3f800000u), denscor));
    s[15][1] = s[0][1];
    s[103][1] = __uint_as_float(0x00000000u);
    s[104][1] = s[82][0];
    s[7][1] = __uint_as_float(0xc0800000u);
    s[8][1] = __uint_as_float(0xc0400000u);
    s[10][1] = pl_fsub(s[1][1], s[39][1]);
    s[58][1] = s[59][0];
    s[9][1] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][1]), s[1][1]);
    xwater = pl_fdiv(water, pl_fmul(pl_fmul(s[0][1], s[88][0]), s[9][1]));
    s[2][1] = pl_fadd(xwater, s[40][1]);
    es = wrf_esat_pr(s, s[1][1]);
    s[11][1] = es;
    s[12][1] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][1], es));
    if ((s[2][1] > s[12][1])) {
        s[3][1] = pl_fadd(pl_fsub(s[2][1], s[12][1]), s[3][1]);
        s[2][1] = s[12][1];
    }
    wrf_waterbal(s);
    return;
}

// chem/module_chem_plumerise_scalar.F:972-1029.
__device__ void wrf_initial(float s[][202], int kmt) {
    int isub = 0;
    int k = 0;
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int lbuoy = 0;
    int itmp = 0;
    int isubm1 = 0;
    float xn1 = 0;
    float xi = 0;
    float es = 0;
    float esat = 0;
    s[61][0] = kmt;
    for (k=1; k<=int(s[61][0]); k+=1) {
        s[10][k] = __uint_as_float(0x00000000u);
        s[0][k] = __uint_as_float(0x00000000u);
        s[1][k] = s[39][k];
        s[15][k] = __uint_as_float(0x00000000u);
        s[16][k] = __uint_as_float(0x00000000u);
        s[2][k] = s[40][k];
        s[7][k] = __uint_as_float(0x00000000u);
        s[8][k] = __uint_as_float(0x00000000u);
        s[4][k] = __uint_as_float(0x00000000u);
        s[5][k] = __uint_as_float(0x00000000u);
        s[3][k] = __uint_as_float(0x00000000u);
        es = wrf_esat_pr(s, s[1][k]);
        s[11][k] = es;
        s[12][k] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][k], es));
        s[9][k] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][k]), s[1][k]);
        s[103][k] = __uint_as_float(0x00000000u);
        s[104][k] = __uint_as_float(0x00000000u);
    }
    s[84][1] = s[82][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[84][k] = pl_fadd(s[84][(k - 1)], pl_fmul(pl_fmul(__uint_as_float(0x3f99999au), s[83][0]), pl_fsub(s[26][k], s[26][(k - 1)])));
    }
    s[84][1] = s[82][0];
    s[104][1] = s[82][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[84][k] = pl_fadd(s[84][(k - 1)], pl_fmul(pl_fmul(__uint_as_float(0x3f99999au), s[83][0]), pl_fsub(s[26][k], s[26][(k - 1)])));
        s[104][k] = s[84][k];
    }
    s[58][1] = s[59][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[58][k] = pl_fmax(__uint_as_float(0x3a83126fu), pl_fsub(s[58][(k - 1)], pl_fdiv(pl_fmul(__uint_as_float(0x3f800000u), s[59][0]), __uint_as_float(0x43480000u))));
    }
    wrf_lbound(s);
    return;
}

// chem/module_chem_plumerise_scalar.F:1034-1050.
__device__ void wrf_damp_grav_wave(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* w, float* t, float* tt, float* qv, float* qh, float* qi, float* qc, float* te, float* pe, float* qvenv) {
    float* dummy = s[121];  // workspace row 121: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dummy[_z] = 0.0f;
    int _a = 0;
    if ((ifrom == 1)) {
        wrf_friction(s, ifrom, nm1, deltak, dt, zt, zm, t, tt, te);
        return;
    }
    for (int _a=1; _a<=200; ++_a) dummy[_a] = __uint_as_float(0x00000000u);
    if ((ifrom == 2)) {
        wrf_friction(s, ifrom, nm1, deltak, dt, zt, zm, w, dummy, dummy);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1055-1085.
__device__ void wrf_friction(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* var1, float* vart, float* var2) {
    int k = 0;
    int nfpt = 0;
    int kf = 0;
    float zmkf = 0;
    float ztop = 0;
    float distim = 0;
    float c1 = 0;
    float c2 = 0;
    kf = (nm1 - int(deltak));
    zmkf = zm[kf];
    ztop = zm[nm1];
    distim = pl_fmin(pl_fmul(__uint_as_float(0x40400000u), dt), __uint_as_float(0x42700000u));
    c1 = pl_fdiv(__uint_as_float(0x3f800000u), pl_fmul(distim, pl_fsub(ztop, zmkf)));
    c2 = pl_fmul(dt, c1);
    if ((ifrom == 1)) {
        for (k=nm1; k>=2; k+=-1) {
            if ((zt[k] <= zmkf)) {
                continue;
            }
            vart[k] = pl_fadd(vart[k], pl_fmul(pl_fmul(c1, pl_fsub(zt[k], zmkf)), pl_fsub(var2[k], var1[k])));
        }
    } else if ((ifrom == 2)) {
        for (k=nm1; k>=2; k+=-1) {
            if ((zt[k] <= zmkf)) {
                continue;
            }
            var1[k] = pl_fadd(var1[k], pl_fmul(pl_fmul(c2, pl_fsub(zt[k], zmkf)), pl_fsub(var2[k], var1[k])));
        }
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1091-1121.
__device__ void wrf_vel_advectc_plumerise(float s[][202], int m1, float* wc, float* wt, float* rho, float* dzm) {
    int k = 0;
    float* flxw = s[122];  // workspace row 122: was a 202-word local array
    for (int _z=0; _z<202; ++_z) flxw[_z] = 0.0f;
    float* dn0 = s[123];  // workspace row 123: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dn0[_z] = 0.0f;
    float c1z = 0;
    int _a = 0;
    for (int _a=1; _a<=m1; ++_a) dn0[_a] = pl_fmul(rho[_a], __uint_as_float(0x3a83126fu));
    flxw[1] = pl_fmul(wc[1], dn0[1]);
    for (k=2; k<=(m1 - 1); k+=1) {
        flxw[k] = pl_fmul(pl_fmul(wc[k], __uint_as_float(0x3f000000u)), pl_fadd(dn0[k], dn0[(k + 1)]));
    }
    c1z = __uint_as_float(0x3f000000u);
    for (k=2; k<=(m1 - 2); k+=1) {
        wt[k] = pl_fadd(wt[k], pl_fmul(pl_fdiv(pl_fmul(c1z, dzm[k]), pl_fadd(dn0[k], dn0[(k + 1)])), pl_fadd(pl_fsub(pl_fmul(pl_fadd(flxw[k], flxw[(k - 1)]), pl_fadd(wc[k], wc[(k - 1)])), pl_fmul(pl_fadd(flxw[k], flxw[(k + 1)]), pl_fadd(wc[k], wc[(k + 1)]))), pl_fmul(pl_fmul(pl_fsub(flxw[(k + 1)], flxw[(k - 1)]), __uint_as_float(0x40000000u)), wc[k]))));
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1127-1147.
__device__ void wrf_hadvance_plumerise(float s[][202], int iac, int m1, float dt, float* wc, float* wt, float* wp, int mintime) {
    int k = 0;
    float* dummy = s[124];  // workspace row 124: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dummy[_z] = 0.0f;
    float eps = 0;
    eps = __uint_as_float(0x3e4ccccdu);
    if ((mintime == 1)) {
        eps = __uint_as_float(0x3f000000u);
    }
    wrf_predict_plumerise(s, m1, wc, wp, wt, dummy, iac, pl_fmul(__uint_as_float(0x40000000u), dt), eps);
    return;
}

// chem/module_chem_plumerise_scalar.F:1152-1193.
__device__ void wrf_predict_plumerise(float s[][202], int npts, float* ac, float* ap, float* fa, float* af, int iac, float dtlp, float epsu) {
    int m = 0;
    if ((iac == 1)) {
        for (m=1; m<=npts; m+=1) {
            ac[m] = pl_fadd(ac[m], pl_fmul(epsu, pl_fsub(ap[m], pl_fmul(__uint_as_float(0x40000000u), ac[m]))));
        }
        return;
    } else if ((iac == 2)) {
        for (m=1; m<=npts; m+=1) {
            af[m] = ap[m];
            ap[m] = pl_fadd(ac[m], pl_fmul(epsu, af[m]));
        }
    }
    for (m=1; m<=npts; m+=1) {
        ac[m] = af[m];
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1198-1228.
__device__ void wrf_buoyancy_plumerise(float s[][202], int m1, float* t, float* te, float* qv, float* qvenv, float* qh, float* qi, float* qc, float* wt, float* scr1) {
    int k = 0;
    float tv = 0;
    float tve = 0;
    float qwtotl = 0;
    float umgamai = 0;
    umgamai = __uint_as_float(0x3f2aaaabu);
    for (k=2; k<=(m1 - 1); k+=1) {
        tv = pl_fdiv(pl_fmul(t[k], pl_fadd(__uint_as_float(0x3f800000u), pl_fdiv(qv[k], __uint_as_float(0x3f1f3b64u)))), pl_fadd(__uint_as_float(0x3f800000u), qv[k]));
        tve = pl_fdiv(pl_fmul(te[k], pl_fadd(__uint_as_float(0x3f800000u), pl_fdiv(qvenv[k], __uint_as_float(0x3f1f3b64u)))), pl_fadd(__uint_as_float(0x3f800000u), qvenv[k]));
        qwtotl = pl_fadd(pl_fadd(qh[k], qi[k]), qc[k]);
        scr1[k] = pl_fmul(pl_fmul(__uint_as_float(0x411ccccdu), umgamai), pl_fsub(pl_fdiv(pl_fsub(tv, tve), tve), qwtotl));
    }
    for (k=2; k<=(m1 - 2); k+=1) {
        wt[k] = pl_fadd(wt[k], pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(scr1[k], scr1[(k + 1)])));
    }
}

// chem/module_chem_plumerise_scalar.F:1234-1273.
__device__ void wrf_entrainment(float s[][202], int m1, float* w, float* wt, float* radius, float alpha) {
    int k = 0;
    float dmdtm = 0;
    float wbar = 0;
    float radius_bar = 0;
    float umgamai = 0;
    float dyn_entr = 0;
    umgamai = __uint_as_float(0x3f2aaaabu);
    for (k=2; k<=(m1 - 1); k+=1) {
        wbar = w[k];
        radius_bar = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(radius[k], radius[(k - 1)]));
        dmdtm = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(umgamai, __uint_as_float(0x40000000u)), alpha), pl_fabs(wbar)), radius_bar);
        wt[k] = pl_fsub(wt[k], pl_fmul(dmdtm, pl_fabs(wbar)));
        dyn_entr = pl_fdiv(pl_fmul(__uint_as_float(0x3ea2f96bu), pl_fabs(pl_fsub(pl_fadd(pl_fsub(s[103][k], s[102][k]), s[103][(k - 1)]), s[102][(k - 1)]))), radius_bar);
        wt[k] = pl_fsub(wt[k], pl_fmul(dyn_entr, pl_fabs(wbar)));
    }
}

// chem/module_chem_plumerise_scalar.F:1279-1401.
__device__ void wrf_scl_advectc_plumerise(float s[][202], char varn, int mzp) {
    float dtlto2 = 0;
    int k = 0;
    int _a = 0;
    dtlto2 = pl_fmul(__uint_as_float(0x3f000000u), s[88][0]);
    s[29][1] = pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][1], s[15][1]), dtlto2), s[9][1]), __uint_as_float(0x3a83126fu));
    s[30][1] = pl_fmul(pl_fmul(pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][1], s[15][1])), dtlto2), s[23][1]);
    for (k=2; k<=mzp; k+=1) {
        s[29][k] = pl_fmul(pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][k], s[15][k]), dtlto2), __uint_as_float(0x3f000000u)), pl_fadd(s[9][k], s[9][(k + 1)])), __uint_as_float(0x3a83126fu));
        s[30][k] = pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][k], s[15][k]), dtlto2), __uint_as_float(0x3f000000u)), s[23][k]);
    }
    for (k=1; k<=mzp; k+=1) {
        s[27][k] = pl_fmul(pl_fsub(s[26][(k + 1)], s[25][k]), s[23][k]);
        s[28][k] = pl_fmul(pl_fsub(s[25][k], s[26][k]), s[23][k]);
        s[31][k] = pl_fdiv(s[24][k], pl_fmul(s[9][k], __uint_as_float(0x3a83126fu)));
    }
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[1][_a];
    wrf_fa_zc_plumerise(s, mzp, s[1], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[1], s[33], s[17], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[2][_a];
    wrf_fa_zc_plumerise(s, mzp, s[2], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[2], s[33], s[18], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[3][_a];
    wrf_fa_zc_plumerise(s, mzp, s[3], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[3], s[33], s[19], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[5][_a];
    wrf_fa_zc_plumerise(s, mzp, s[5], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[5], s[33], s[21], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[4][_a];
    wrf_fa_zc_plumerise(s, mzp, s[4], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[4], s[33], s[20], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[103][_a];
    wrf_fa_zc_plumerise(s, mzp, s[103], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[103], s[33], s[105], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[104][_a];
    wrf_fa_zc_plumerise(s, mzp, s[104], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[104], s[33], s[106], s[88][0]);
    return;
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[6][_a];
    wrf_fa_zc_plumerise(s, mzp, s[6], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    wrf_advtndc_plumerise(s, mzp, s[6], s[33], s[22], s[88][0]);
    return;
}

// chem/module_chem_plumerise_scalar.F:1407-1449.
__device__ void wrf_fa_zc_plumerise(float s[][202], int m1, float* scp, float* scr1, float* vt3dc, float* vt3df, float* vt3dg, float* vt3dk, float* vctr1, float* vctr2) {
    int k = 0;
    float dfact = 0;
    dfact = __uint_as_float(0x3f000000u);
    for (k=1; k<=(m1 - 1); k+=1) {
        vt3dg[k] = pl_fmul(vt3dc[k], pl_fadd(pl_fadd(pl_fmul(vctr1[k], scr1[k]), pl_fmul(vctr2[k], scr1[(k + 1)])), pl_fmul(vt3df[k], pl_fsub(scr1[k], scr1[(k + 1)]))));
    }
    for (k=1; k<=(m1 - 1); k+=1) {
        if ((vt3dc[k] > __uint_as_float(0x00000000u))) {
            if ((pl_fmul(vt3dg[k], vt3dk[k]) > pl_fmul(dfact, scr1[k]))) {
                vt3dg[k] = pl_fmul(vt3dc[k], scr1[k]);
            }
        } else if ((vt3dc[k] < __uint_as_float(0x00000000u))) {
            if ((pl_fmul((-vt3dg[k]), vt3dk[(k + 1)]) > pl_fmul(dfact, scr1[(k + 1)]))) {
                vt3dg[k] = pl_fmul(vt3dc[k], scr1[(k + 1)]);
            }
        }
    }
    for (k=2; k<=(m1 - 1); k+=1) {
        scr1[k] = pl_fadd(scr1[k], pl_fmul(vt3dk[k], pl_fadd(pl_fsub(vt3dg[(k - 1)], vt3dg[k]), pl_fmul(scp[k], pl_fsub(vt3dc[k], vt3dc[(k - 1)])))));
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1454-1463.
__device__ void wrf_advtndc_plumerise(float s[][202], int m1, float* scp, float* sca, float* sct, float dtl) {
    int k = 0;
    float dtli = 0;
    dtli = pl_fdiv(__uint_as_float(0x3f800000u), dtl);
    for (k=2; k<=(m1 - 1); k+=1) {
        sct[k] = pl_fadd(sct[k], pl_fmul(pl_fsub(sca[k], scp[k]), dtli));
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:1469-1476.
__device__ void wrf_tend0_plumerise(float s[][202]) {
    int _a = 0;
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[16][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[17][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[18][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[19][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[20][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[21][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[105][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[106][_a] = __uint_as_float(0x00000000u);
}

// chem/module_chem_plumerise_scalar.F:1484-1515.
__device__ void wrf_scl_misc(float s[][202], int m1) {
    int k = 0;
    float dmdtm = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        s[73][0] = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][k], s[0][(k - 1)]));
        s[72][0] = pl_fdiv(pl_fmul((-s[73][0]), __uint_as_float(0x411cf5c3u)), __uint_as_float(0x447b0000u));
        dmdtm = pl_fdiv(pl_fmul(pl_fmul(__uint_as_float(0x40000000u), s[83][0]), pl_fabs(s[73][0])), s[84][k]);
        s[17][k] = pl_fsub(pl_fadd(s[17][k], s[72][0]), pl_fmul(dmdtm, pl_fsub(s[1][k], s[39][k])));
        s[18][k] = pl_fsub(s[18][k], pl_fmul(dmdtm, pl_fsub(s[2][k], s[40][k])));
        s[19][k] = pl_fsub(s[19][k], pl_fmul(dmdtm, s[3][k]));
        s[20][k] = pl_fsub(s[20][k], pl_fmul(dmdtm, s[4][k]));
        s[21][k] = pl_fsub(s[21][k], pl_fmul(dmdtm, s[5][k]));
        s[105][k] = pl_fsub(s[105][k], pl_fmul(dmdtm, pl_fsub(s[103][k], s[102][k])));
        s[106][k] = pl_fadd(s[106][k], pl_fmul(pl_fmul(pl_fmul(__uint_as_float(0x3f000000u), dmdtm), __uint_as_float(0x3f99999au)), s[84][k]));
    }
}

// chem/module_chem_plumerise_scalar.F:1521-1584.
__device__ void wrf_scl_dyn_entrain(float s[][202], int m1, int nkp, float wbar, float* w, float adiabat, float alpha, float* radius, float* tt, float* t, float* te, float* qvt, float* qv, float* qvenv, float* qct, float* qc, float* qht, float* qh, float* qit, float* qi, float* vel_e, float* vel_p, float* vel_t, float* rad_p, float* rad_t) {
    int k = 0;
    float dmdtm = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        rad_t[k] = pl_fadd(rad_t[k], pl_fdiv(pl_fabs(pl_fsub(vel_e[k], vel_p[k])), __uint_as_float(0x40490ff9u)));
        dmdtm = pl_fdiv(pl_fmul(__uint_as_float(0x3f22f96bu), pl_fabs(pl_fsub(vel_e[k], vel_p[k]))), radius[k]);
        vel_t[k] = pl_fsub(vel_t[k], pl_fmul(dmdtm, pl_fsub(vel_p[k], vel_e[k])));
        tt[k] = pl_fsub(tt[k], pl_fmul(dmdtm, pl_fsub(t[k], te[k])));
        qvt[k] = pl_fsub(qvt[k], pl_fmul(dmdtm, pl_fsub(qv[k], qvenv[k])));
        qct[k] = pl_fsub(qct[k], pl_fmul(dmdtm, qc[k]));
        qht[k] = pl_fsub(qht[k], pl_fmul(dmdtm, qh[k]));
        qit[k] = pl_fsub(qit[k], pl_fmul(dmdtm, qi[k]));
    }
}

// chem/module_chem_plumerise_scalar.F:1591-1627.
__device__ void wrf_visc_w(float s[][202], int m1, int deltak, int kmt) {
    int k = 0;
    int m2 = 0;
    float dz1t = 0;
    float dz1m = 0;
    float dz2t = 0;
    float dz2m = 0;
    float d2wdz = 0;
    float d2tdz = 0;
    float d2qvdz = 0;
    float d2qhdz = 0;
    float d2qcdz = 0;
    float d2qidz = 0;
    float d2scdz = 0;
    float d2vel_pdz = 0;
    float d2rad_dz = 0;
    m2 = min(m1, kmt);
    for (k=2; k<=(m2 - 1); k+=1) {
        dz1t = pl_fmul(__uint_as_float(0x3f000000u), pl_fsub(s[26][(k + 1)], s[26][(k - 1)]));
        dz2t = pl_fdiv(s[58][k], pl_fmul(dz1t, dz1t));
        dz1m = pl_fmul(__uint_as_float(0x3f000000u), pl_fsub(s[25][(k + 1)], s[25][(k - 1)]));
        dz2m = pl_fdiv(s[58][k], pl_fmul(dz1m, dz1m));
        d2wdz = pl_fmul(pl_fadd(pl_fsub(s[0][(k + 1)], pl_fmul(2, s[0][k])), s[0][(k - 1)]), dz2m);
        d2tdz = pl_fmul(pl_fadd(pl_fsub(s[1][(k + 1)], pl_fmul(2, s[1][k])), s[1][(k - 1)]), dz2t);
        d2qvdz = pl_fmul(pl_fadd(pl_fsub(s[2][(k + 1)], pl_fmul(2, s[2][k])), s[2][(k - 1)]), dz2t);
        d2qhdz = pl_fmul(pl_fadd(pl_fsub(s[4][(k + 1)], pl_fmul(2, s[4][k])), s[4][(k - 1)]), dz2t);
        d2qcdz = pl_fmul(pl_fadd(pl_fsub(s[3][(k + 1)], pl_fmul(2, s[3][k])), s[3][(k - 1)]), dz2t);
        d2qidz = pl_fmul(pl_fadd(pl_fsub(s[5][(k + 1)], pl_fmul(2, s[5][k])), s[5][(k - 1)]), dz2t);
        d2vel_pdz = pl_fmul(pl_fadd(pl_fsub(s[103][(k + 1)], pl_fmul(2, s[103][k])), s[103][(k - 1)]), dz2t);
        d2rad_dz = pl_fmul(pl_fadd(pl_fsub(s[104][(k + 1)], pl_fmul(2, s[104][k])), s[104][(k - 1)]), dz2t);
        s[16][k] = pl_fadd(s[16][k], d2wdz);
        s[17][k] = pl_fadd(s[17][k], d2tdz);
        s[18][k] = pl_fadd(s[18][k], d2qvdz);
        s[19][k] = pl_fadd(s[19][k], d2qcdz);
        s[20][k] = pl_fadd(s[20][k], d2qhdz);
        s[21][k] = pl_fadd(s[21][k], d2qidz);
        s[105][k] = pl_fadd(s[105][k], d2vel_pdz);
        s[106][k] = pl_fadd(s[106][k], d2rad_dz);
    }
}

// chem/module_chem_plumerise_scalar.F:1635-1667.
__device__ void wrf_update_plumerise(float s[][202], int m1, char varn) {
    int k = 0;
    if ((varn == 'w')) {
        for (k=2; k<=(m1 - 1); k+=1) {
            s[0][k] = pl_fadd(s[0][k], pl_fmul(s[16][k], s[88][0]));
        }
        return;
    } else {
        for (k=2; k<=(m1 - 1); k+=1) {
            s[1][k] = pl_fadd(s[1][k], pl_fmul(s[17][k], s[88][0]));
            s[2][k] = pl_fadd(s[2][k], pl_fmul(s[18][k], s[88][0]));
            s[3][k] = pl_fadd(s[3][k], pl_fmul(s[19][k], s[88][0]));
            s[4][k] = pl_fadd(s[4][k], pl_fmul(s[20][k], s[88][0]));
            s[5][k] = pl_fadd(s[5][k], pl_fmul(s[21][k], s[88][0]));
            s[2][k] = pl_fmax(__uint_as_float(0x00000000u), s[2][k]);
            s[3][k] = pl_fmax(__uint_as_float(0x00000000u), s[3][k]);
            s[4][k] = pl_fmax(__uint_as_float(0x00000000u), s[4][k]);
            s[5][k] = pl_fmax(__uint_as_float(0x00000000u), s[5][k]);
            s[103][k] = pl_fadd(s[103][k], pl_fmul(s[105][k], s[88][0]));
            s[104][k] = pl_fadd(s[104][k], pl_fmul(s[106][k], s[88][0]));
        }
    }
}

// chem/module_chem_plumerise_scalar.F:1673-1729.
__device__ void wrf_fallpart(float s[][202], int m1) {
    int k = 0;
    float vtc = 0;
    float dfhz = 0;
    float dfiz = 0;
    float dz1 = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        vtc = pl_fmul(__uint_as_float(0x40a36fb7u), pl_fpow(s[9][k], __uint_as_float(0x3e000000u)));
        s[7][k] = __uint_as_float(0xc0800000u);
        s[75][0] = pl_fadd(s[0][k], s[7][k]);
        s[70][k] = pl_fadd(__uint_as_float(0x3fcccccdu), pl_fmul(__uint_as_float(0x3a156c0du), pl_fpow(pl_fabs(s[75][0]), __uint_as_float(0x3fc00000u))));
        s[8][k] = __uint_as_float(0xc0400000u);
        s[76][0] = pl_fadd(s[0][k], s[8][k]);
        s[71][k] = pl_fadd(__uint_as_float(0x3fcccccdu), pl_fdiv(pl_fmul(__uint_as_float(0x3a156c0du), pl_fpow(pl_fabs(s[76][0]), __uint_as_float(0x3fc00000u))), __uint_as_float(0x3f400000u)));
        if ((s[75][0] >= __uint_as_float(0x00000000u))) {
            dfhz = pl_fdiv(pl_fmul(s[4][k], pl_fsub(pl_fmul(s[9][k], s[7][k]), pl_fmul(s[9][(k - 1)], s[7][(k - 1)]))), s[9][(k - 1)]);
        } else {
            dfhz = pl_fdiv(pl_fmul(s[4][k], pl_fsub(pl_fmul(s[9][(k + 1)], s[7][(k + 1)]), pl_fmul(s[9][k], s[7][k]))), s[9][k]);
        }
        if ((s[76][0] >= __uint_as_float(0x00000000u))) {
            dfiz = pl_fdiv(pl_fmul(s[5][k], pl_fsub(pl_fmul(s[9][k], s[8][k]), pl_fmul(s[9][(k - 1)], s[8][(k - 1)]))), s[9][(k - 1)]);
        } else {
            dfiz = pl_fdiv(pl_fmul(s[5][k], pl_fsub(pl_fmul(s[9][(k + 1)], s[8][(k + 1)]), pl_fmul(s[9][k], s[8][k]))), s[9][k]);
        }
        dz1 = pl_fsub(s[25][k], s[25][(k - 1)]);
        s[20][k] = pl_fsub(s[20][k], pl_fdiv(dfhz, dz1));
        s[21][k] = pl_fsub(s[21][k], pl_fdiv(dfiz, dz1));
    }
}

// chem/module_chem_plumerise_scalar.F:1818-1835.
__device__ void wrf_waterbal(float s[][202]) {
    if ((s[3][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    if ((s[4][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    if ((s[5][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    wrf_evaporate(s);
    wrf_sublimate(s);
    wrf_glaciate(s);
    wrf_melt(s);
    wrf_convert(s);
    return;
}

// chem/module_chem_plumerise_scalar.F:1843-2048.
__device__ void wrf_evaporate(float s[][202]) {
    float evhdt = 0;
    float evidt = 0;
    float evrate = 0;
    float evap = 0;
    float sd = 0;
    float quant = 0;
    float dividend = 0;
    float divisor = 0;
    float devidt = 0;
    sd = pl_fsub(s[12][int(s[63][0])], s[2][int(s[63][0])]);
    if ((sd == __uint_as_float(0x00000000u))) {
        return;
    }
    evhdt = __uint_as_float(0x00000000u);
    evidt = __uint_as_float(0x00000000u);
    evrate = pl_fabs(pl_fmul(s[73][0], s[57][0]));
    evap = pl_fmul(evrate, s[88][0]);
    if ((sd <= __uint_as_float(0x00000000u))) {
        if ((evap >= pl_fabs(sd))) {
            s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
            s[2][int(s[63][0])] = s[12][int(s[63][0])];
            s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
            return;
        } else {
            s[3][int(s[63][0])] = pl_fadd(s[3][int(s[63][0])], evap);
            s[2][int(s[63][0])] = pl_fsub(s[2][int(s[63][0])], evap);
            s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(evap, __uint_as_float(0x451ba0a4u)));
            return;
        }
    } else {
        if ((evap <= s[3][int(s[63][0])])) {
            if ((sd <= evap)) {
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                return;
            } else {
                sd = pl_fsub(sd, evap);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evap);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evap, __uint_as_float(0x451ba0a4u)));
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], evap);
            }
        } else {
            if ((sd <= s[3][int(s[63][0])])) {
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                return;
            } else {
                sd = pl_fsub(sd, s[3][int(s[63][0])]);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[3][int(s[63][0])]);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[3][int(s[63][0])], __uint_as_float(0x451ba0a4u)));
                s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
            }
        }
        if ((s[4][int(s[63][0])] > __uint_as_float(0x2edbe6ffu))) {
            quant = pl_fmul(pl_fsub(pl_fsub(s[12][int(s[63][0])], s[3][int(s[63][0])]), s[2][int(s[63][0])]), s[9][int(s[63][0])]);
            evhdt = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(s[88][0], __uint_as_float(0x3a0e9b39u)), quant), pl_fpow(pl_fmul(s[4][int(s[63][0])], s[9][int(s[63][0])]), __uint_as_float(0x3f266666u))), s[9][int(s[63][0])]);
            if ((evhdt <= s[4][int(s[63][0])])) {
                if ((sd <= evhdt)) {
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], sd);
                    s[2][int(s[63][0])] = s[12][int(s[63][0])];
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                    return;
                } else {
                    sd = pl_fsub(sd, evhdt);
                    s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evhdt);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evhdt, __uint_as_float(0x451ba0a4u)));
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], evhdt);
                }
            } else {
                if ((sd <= s[4][int(s[63][0])])) {
                    s[2][int(s[63][0])] = s[12][int(s[63][0])];
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], sd);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                    return;
                } else {
                    sd = pl_fsub(sd, s[4][int(s[63][0])]);
                    s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[4][int(s[63][0])]);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[4][int(s[63][0])], __uint_as_float(0x451ba0a4u)));
                    s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
                }
            }
        }
        if ((s[5][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
            return;
        }
        dividend = pl_fmul(pl_fmul(pl_fmul(pl_fpow(pl_fdiv(__uint_as_float(0x49742400u), s[9][int(s[63][0])]), __uint_as_float(0x3ef33333u)), pl_fsub(pl_fdiv(sd, s[12][int(s[63][0])]), 1)), pl_fpow(s[5][int(s[63][0])], __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90a3d7u));
        divisor = pl_fadd(__uint_as_float(0x492ae600u), pl_fdiv(__uint_as_float(0x4a7a3e80u), pl_fmul(__uint_as_float(0x41200000u), s[11][int(s[63][0])])));
        devidt = pl_fdiv(pl_fmul((-s[71][int(s[63][0])]), dividend), divisor);
        evidt = pl_fmul(devidt, s[88][0]);
        if ((evidt <= s[5][int(s[63][0])])) {
            if ((sd <= evidt)) {
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], sd);
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x45306b59u)));
                return;
            } else {
                sd = pl_fsub(sd, evidt);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evidt);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evidt, __uint_as_float(0x45306b59u)));
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], evidt);
            }
        } else {
            if ((sd <= s[5][int(s[63][0])])) {
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], sd);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x45306b59u)));
                return;
            } else {
                sd = pl_fsub(sd, s[5][int(s[63][0])]);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[5][int(s[63][0])]);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[5][int(s[63][0])], __uint_as_float(0x45306b59u)));
                s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
            }
        }
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2059-2122.
__device__ void wrf_convert(float s[][202]) {
    float accrete = 0;
    float con = 0;
    float q = 0;
    float h = 0;
    float bc1 = 0;
    float bc2 = 0;
    float total = 0;
    if ((s[1][int(s[63][0])] <= __uint_as_float(0x4386a666u))) {
        return;
    }
    if ((s[3][int(s[63][0])] == __uint_as_float(0x00000000u))) {
        return;
    }
    accrete = __uint_as_float(0x00000000u);
    con = __uint_as_float(0x00000000u);
    q = pl_fmul(s[9][int(s[63][0])], s[3][int(s[63][0])]);
    h = pl_fmul(s[9][int(s[63][0])], s[4][int(s[63][0])]);
    if ((s[4][int(s[63][0])] > __uint_as_float(0x00000000u))) {
        accrete = pl_fmul(pl_fmul(__uint_as_float(0x3baa64c3u), q), pl_fpow(h, __uint_as_float(0x3f600000u)));
    }
    if ((1 != 0)) {
        con = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(q, q), q), __uint_as_float(0x3e158106u)), pl_fmul(__uint_as_float(0x42700000u), pl_fadd(pl_fmul(pl_fmul(__uint_as_float(0x40a00000u), q), __uint_as_float(0x3e158106u)), __uint_as_float(0x4564c000u))));
    } else {
        con = pl_fmax(__uint_as_float(0x00000000u), pl_fmul(__uint_as_float(0x3a83126fu), pl_fsub(q, __uint_as_float(0x3f000000u))));
    }
    total = pl_fdiv(pl_fmul(pl_fadd(con, accrete), s[88][0]), s[9][int(s[63][0])]);
    if ((total < s[3][int(s[63][0])])) {
        s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], total);
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], total);
        return;
    } else {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], s[3][int(s[63][0])]);
        s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2130-2232.
__device__ void wrf_convert2(float s[][202]) {
    float ka = 0;
    float keins = 0;
    float kzwei = 0;
    float kdrei = 0;
    float vt = 0;
    float a = 0;
    float b = 0;
    float c = 0;
    float d = 0;
    float con = 0;
    float accrete = 0;
    float total = 0;
    float* y = s[125];  // workspace row 125: was a 202-word local array
    for (int _z=0; _z<202; ++_z) y[_z] = 0.0f;
    float roh = 0;
    a = __uint_as_float(0x00000000u);
    b = __uint_as_float(0x00000000u);
    y[1] = s[1][int(s[63][0])];
    y[4] = s[0][int(s[63][0])];
    y[2] = s[3][int(s[63][0])];
    y[3] = s[4][int(s[63][0])];
    y[5] = s[84][int(s[63][0])];
    roh = pl_fmul(s[9][int(s[63][0])], __uint_as_float(0x3a83126fu));
    ka = __uint_as_float(0x3a03126fu);
    if ((y[1] < __uint_as_float(0x43811333u))) {
        keins = __uint_as_float(0x3a6bedfau);
        kzwei = __uint_as_float(0x3baa64c3u);
        kdrei = __uint_as_float(0x41763d71u);
    } else {
        keins = __uint_as_float(0x3ac49ba6u);
        kzwei = __uint_as_float(0x3be410b6u);
        kdrei = __uint_as_float(0x413947aeu);
    }
    vt = pl_fmul((-kdrei), pl_fpow(pl_fdiv(y[3], roh), __uint_as_float(0x3e000000u)));
    if ((y[4] > __uint_as_float(0x00000000u))) {
        if (true) {
            a = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(pl_fdiv(1, y[4]), y[2]), y[2]), __uint_as_float(0x447a0000u)), pl_fmul(__uint_as_float(0x42700000u), pl_fadd(__uint_as_float(0x40a00000u), pl_fdiv(__uint_as_float(0x425b999au), pl_fmul(pl_fmul(y[2], __uint_as_float(0x447a0000u)), __uint_as_float(0x3f800000u))))));
        } else {
            if ((y[2] > pl_fmul(ka, roh))) {
                a = pl_fmul(pl_fdiv(keins, y[4]), pl_fsub(y[2], pl_fmul(ka, roh)));
            }
        }
    } else {
        a = __uint_as_float(0x00000000u);
    }
    if ((y[4] > __uint_as_float(0x00000000u))) {
        b = pl_fmul(pl_fmul(pl_fmul(pl_fdiv(kzwei, pl_fsub(y[4], vt)), pl_fmax(__uint_as_float(0x00000000u), y[2])), pl_fpow(pl_fmax(__uint_as_float(0x3a83126fu), roh), __uint_as_float(0xbf600000u))), pl_fpow(pl_fmax(__uint_as_float(0x00000000u), y[3]), __uint_as_float(0x3f600000u)));
    } else {
        b = __uint_as_float(0x00000000u);
    }
    con = a;
    accrete = b;
    total = pl_fdiv(pl_fmul(pl_fadd(con, accrete), pl_fdiv(1, s[23][int(s[63][0])])), roh);
    if ((total < s[3][int(s[63][0])])) {
        s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], total);
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], total);
        return;
    } else {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], s[3][int(s[63][0])]);
        s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2249-2292.
__device__ void wrf_sublimate(float s[][202]) {
    float dtsubh = 0;
    float dividend = 0;
    float divisor = 0;
    float subl = 0;
    dtsubh = __uint_as_float(0x00000000u);
    if ((s[1][int(s[63][0])] > __uint_as_float(0x4386a666u))) {
        return;
    }
    if ((s[2][int(s[63][0])] <= s[12][int(s[63][0])])) {
        return;
    }
    dividend = pl_fmul(pl_fmul(pl_fmul(pl_fpow(pl_fdiv(__uint_as_float(0x49742400u), s[9][int(s[63][0])]), __uint_as_float(0x3ef33333u)), pl_fsub(pl_fdiv(s[2][int(s[63][0])], s[12][int(s[63][0])]), 1)), pl_fpow(s[5][int(s[63][0])], __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90a3d7u));
    divisor = pl_fadd(__uint_as_float(0x492ae600u), pl_fdiv(__uint_as_float(0x4a7a3e80u), pl_fmul(__uint_as_float(0x41200000u), s[11][int(s[63][0])])));
    dtsubh = pl_fabs(pl_fdiv(dividend, divisor));
    subl = pl_fmul(dtsubh, s[88][0]);
    if ((subl < s[2][int(s[63][0])])) {
        s[2][int(s[63][0])] = pl_fsub(s[2][int(s[63][0])], subl);
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], subl);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(subl, __uint_as_float(0x45306b59u)));
        return;
    } else {
        s[5][int(s[63][0])] = s[2][int(s[63][0])];
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(s[2][int(s[63][0])], __uint_as_float(0x45306b59u)));
        s[2][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2305-2345.
__device__ void wrf_glaciate(float s[][202]) {
    float dfrzh = 0;
    dfrzh = __uint_as_float(0x00000000u);
    if ((s[4][int(s[63][0])] <= __uint_as_float(0x00000000u))) {
        return;
    }
    if ((s[2][int(s[63][0])] < s[12][int(s[63][0])])) {
        return;
    }
    if ((s[1][int(s[63][0])] > __uint_as_float(0x4386a666u))) {
        return;
    }
    dfrzh = pl_fmul(pl_fmul(s[88][0], __uint_as_float(0x3ccccccdu)), s[4][int(s[63][0])]);
    if ((dfrzh < s[4][int(s[63][0])])) {
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], dfrzh);
        s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], dfrzh);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a655adu), dfrzh));
        return;
    } else {
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], s[4][int(s[63][0])]);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a655adu), s[4][int(s[63][0])]));
        s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2356-2391.
__device__ void wrf_melt(float s[][202]) {
    float dtmelt = 0;
    dtmelt = __uint_as_float(0x00000000u);
    if ((s[5][int(s[63][0])] <= __uint_as_float(0x00000000u))) {
        return;
    }
    if ((s[1][int(s[63][0])] < __uint_as_float(0x43888000u))) {
        return;
    }
    dtmelt = pl_fmul(pl_fmul(pl_fmul(pl_fmul(pl_fmul(s[88][0], pl_fdiv(__uint_as_float(0x401147aeu), s[9][int(s[63][0])])), s[71][int(s[63][0])]), pl_fsub(s[1][int(s[63][0])], __uint_as_float(0x43888000u))), pl_fpow(pl_fmul(pl_fmul(s[9][int(s[63][0])], s[5][int(s[63][0])]), __uint_as_float(0x358637bdu)), __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90705du));
    if ((dtmelt < s[5][int(s[63][0])])) {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], dtmelt);
        s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], dtmelt);
        s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a6228fu), dtmelt));
        return;
    } else {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], s[5][int(s[63][0])]);
        s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a6228fu), s[5][int(s[63][0])]));
        s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// chem/module_chem_plumerise_scalar.F:2396-2433.
__device__ void wrf_htint(float s[][202], int nzz1, float* vctra, float* eleva, int nzz2, float* vctrb, float* elevb) {
    int l = 0;
    int k = 0;
    int kk = 0;
    float wt = 0;
    l = 1;
    for (k=1; k<=nzz2; k+=1) {
        while (true) {
            if (((elevb[k] < eleva[1]) || ((elevb[k] >= eleva[l]) && (elevb[k] <= eleva[(l + 1)])))) {
                wt = pl_fdiv(pl_fsub(elevb[k], eleva[l]), pl_fsub(eleva[(l + 1)], eleva[l]));
                vctrb[k] = pl_fadd(vctra[l], pl_fmul(pl_fsub(vctra[(l + 1)], vctra[l]), wt));
                break;
            } else if ((elevb[k] > eleva[nzz1])) {
                wt = pl_fdiv(pl_fsub(elevb[k], eleva[nzz1]), pl_fsub(eleva[(nzz1 - 1)], eleva[nzz1]));
                vctrb[k] = pl_fadd(vctra[nzz1], pl_fmul(pl_fsub(vctra[(nzz1 - 1)], vctra[nzz1]), wt));
                break;
            }
            l = (l + 1);
            if ((l == nzz1)) {
                for (kk=1; kk<=l; kk+=1) {
                }
                return;
            }
        }
    }
}

// chem/module_chem_plumerise_scalar.F:2440-2462.
__device__ float wrf_esat_pr(float s[][202], float tem) {
    float esat_pr = 0;
    float temc = 0;
    float esatm = 0;
    temc = pl_fsub(tem, __uint_as_float(0x4388a666u));
    if ((temc <= __uint_as_float(0xc2200000u))) {
        esatm = pl_fmul(__uint_as_float(0x40c39168u), pl_fexp(pl_fdiv(pl_fmul(__uint_as_float(0x41b45604u), temc), pl_fadd(temc, __uint_as_float(0x4388bd71u)))));
        esat_pr = pl_fdiv(esatm, __uint_as_float(0x41200000u));
        return esat_pr;
    }
    esatm = pl_fmul(__uint_as_float(0x40c39653u), pl_fexp(pl_fdiv(pl_fmul(pl_fsub(__uint_as_float(0x4195d4feu), pl_fdiv(temc, __uint_as_float(0x43634ccdu))), temc), pl_fadd(temc, __uint_as_float(0x4380ef5cu)))));
    esat_pr = pl_fdiv(esatm, __uint_as_float(0x41200000u));
    return esat_pr;
}

// physics/smoke_dust/module_smoke_plumerise.F90:132-207.
__device__ void gsl_get_env_condition(float s[][202], int k1, int k2, int& kmt, int wind_eff, float g, float cp, float rgas, float cpor) {
    int k = 0;
    int kcon = 0;
    int klcl = 0;
    int nk = 0;
    int nkmid = 0;
    int i = 0;
    float znz = 0;
    float themax = 0;
    float tlll = 0;
    float plll = 0;
    float rlll = 0;
    float zlll = 0;
    float dzdd = 0;
    float dzlll = 0;
    float tlcl = 0;
    float plcl = 0;
    float dzlcl = 0;
    float dummy = 0;
    char errmsg = 0;
    int errflg = 0;
    int _a = 0;
    if ((!bool(s[113][0]))) {
        gsl_set_grid(s);
    }
    znz = s[53][k2];
    ;
    for (k=200; k>=1; k+=-1) {
        if ((s[26][k] < znz)) {
            ;
            break;
        }
    }
    if ((errflg != 0)) {
        ;
        return;
    }
    kmt = min(k, 199);
    nk = ((k2 - k1) + 1);
    gsl_htint(s, nk, s[44], s[53], kmt, s[100], s[26]);
    if ((errflg != 0)) {
        return;
    }
    gsl_htint(s, nk, s[45], s[53], kmt, s[101], s[26]);
    if ((errflg != 0)) {
        return;
    }
    gsl_htint(s, nk, s[47], s[53], kmt, s[35], s[26]);
    if ((errflg != 0)) {
        return;
    }
    gsl_htint(s, nk, s[48], s[53], kmt, s[40], s[26]);
    if ((errflg != 0)) {
        return;
    }
    for (k=1; k<=kmt; k+=1) {
        s[40][k] = pl_fmax(s[40][k], __uint_as_float(0x322bcc77u));
    }
    s[34][1] = s[49][1];
    for (k=1; k<=kmt; k+=1) {
        s[36][k] = pl_fmul(s[35][k], pl_fadd(__uint_as_float(0x3f800000u), pl_fmul(__uint_as_float(0x3f1c28f6u), s[40][k])));
    }
    for (k=2; k<=kmt; k+=1) {
        s[34][k] = pl_fsub(s[34][(k - 1)], pl_fdiv(pl_fmul(pl_fmul(g, __uint_as_float(0x40000000u)), pl_fsub(s[26][k], s[26][(k - 1)])), pl_fadd(s[36][k], s[36][(k - 1)])));
    }
    for (k=1; k<=kmt; k+=1) {
        s[39][k] = pl_fdiv(pl_fmul(s[35][k], s[34][k]), cp);
        s[38][k] = pl_fmul(pl_fpow(pl_fdiv(s[34][k], cp), cpor), __uint_as_float(0x47c35000u));
        s[42][k] = pl_fdiv(s[38][k], pl_fmul(pl_fmul(rgas, s[39][k]), pl_fadd(__uint_as_float(0x3f800000u), pl_fmul(__uint_as_float(0x3f1c28f6u), s[40][k]))));
        s[102][k] = pl_fsqrt(pl_fadd(pl_fpowi(s[100][k], 2), pl_fpowi(s[101][k], 2)));
    }
    if ((wind_eff < 1)) {
        for (int _a=1; _a<=kmt; ++_a) s[102][_a] = __uint_as_float(0x00000000u);
    }
    for (k=1; k<=kmt; k+=1) {
        s[38][k] = pl_fmul(s[38][k], __uint_as_float(0x3a83126fu));
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:214-245.
__device__ void gsl_set_grid(float s[][202]) {
    int k = 0;
    int mzp = 0;
    s[56][0] = __uint_as_float(0x42c80000u);
    mzp = 200;
    s[26][1] = s[77][0];
    s[25][1] = s[77][0];
    s[26][2] = pl_fadd(s[26][1], pl_fmul(__uint_as_float(0x3f000000u), s[56][0]));
    s[25][2] = pl_fadd(s[25][1], s[56][0]);
    for (k=3; k<=mzp; k+=1) {
        s[26][k] = pl_fadd(s[26][(k - 1)], s[56][0]);
        s[25][k] = pl_fadd(s[25][(k - 1)], s[56][0]);
    }
    for (k=1; k<=(mzp - 1); k+=1) {
        s[23][k] = pl_fdiv(__uint_as_float(0x3f800000u), pl_fsub(s[26][(k + 1)], s[26][k]));
    }
    s[23][mzp] = s[23][(mzp - 1)];
    for (k=2; k<=mzp; k+=1) {
        s[24][k] = pl_fdiv(__uint_as_float(0x3f800000u), pl_fsub(s[25][k], s[25][(k - 1)]));
    }
    s[24][1] = pl_fdiv(pl_fmul(s[24][2], s[24][2]), s[24][3]);
    s[113][0] = true;
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:251-283.
__device__ void gsl_set_flam_vert(float s[][202], float* ztopmax, int& k1, int& k2, int nkp, float* zzcon) {
    int imm = 0;
    int k = 0;
    int* k_lim = reinterpret_cast<int*>(s[126]);  // workspace row 126: was a 202-word local array
    for (int _z=0; _z<202; ++_z) k_lim[_z] = 0;
    for (imm=1; imm<=2; imm+=1) {
        for (k=1; k<=(nkp - 1); k+=1) {
            if ((zzcon[k] > ztopmax[imm])) {
                break;
            }
        }
        k_lim[imm] = k;
    }
    k1 = min(max(4, k_lim[1]), 51);
    k2 = min(51, k_lim[2]);
    if ((k2 <= k1)) {
        k2 = (k1 + 1);
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:290-391.
__device__ void gsl_get_fire_properties(float s[][202], int imm, float burnt_area, float frp) {
    int moist = 0;
    int i = 0;
    int icount = 0;
    float bfract = 0;
    float effload = 0;
    float heat = 0;
    float hinc = 0;
    float heat_fluxw = 0;
    int errflg = 0;
    char errmsg = 0;
    s[81][0] = burnt_area;
    heat_fluxw = pl_fdiv(pl_fmul(__uint_as_float(0x3f6147aeu), pl_fdiv(frp, s[81][0])), __uint_as_float(0x3f0ccccdu));
    s[92][0] = 53;
    s[87][0] = __uint_as_float(0x41200000u);
    moist = 10;
    s[93][0] = (int(s[92][0]) + 2);
    heat = __uint_as_float(0x4b933f50u);
    s[93][0] = (int(s[93][0]) * 60);
    s[82][0] = pl_fsqrt(pl_fdiv(s[81][0], __uint_as_float(0x40490fd0u)));
    s[86][0] = pl_fdiv(moist, __uint_as_float(0x42c80000u));
    for (i=1; i<=200; i+=1) {
        s[85][i] = __uint_as_float(0x38d1b717u);
    }
    s[90][0] = pl_fmul(int(s[92][0]), __uint_as_float(0x42700000u));
    bfract = __uint_as_float(0x3f800000u);
    effload = pl_fmul(s[87][0], bfract);
    icount = 1;
    if ((int(s[92][0]) > 200)) {
        ;
        ;
        return;
    }
    while ((icount <= int(s[92][0]))) {
        s[85][icount] = pl_fmul(heat_fluxw, __uint_as_float(0x3f0ccccdu));
        icount = (icount + 1);
    }
    if ((1 != 1)) {
        hinc = pl_fdiv(s[85][1], __uint_as_float(0x40800000u));
        s[85][1] = __uint_as_float(0x3dcccccdu);
        s[85][2] = hinc;
        s[85][3] = pl_fmul(__uint_as_float(0x40000000u), hinc);
        s[85][4] = pl_fmul(__uint_as_float(0x40400000u), hinc);
    } else {
        hinc = pl_fdiv(s[85][1], __uint_as_float(0x40800000u));
        if ((imm == 1)) {
            s[85][1] = __uint_as_float(0x3dcccccdu);
            s[85][2] = hinc;
            s[85][3] = pl_fmul(__uint_as_float(0x40000000u), hinc);
            s[85][4] = pl_fmul(__uint_as_float(0x40400000u), hinc);
        } else {
            s[85][2] = pl_fadd(s[85][1], hinc);
            s[85][3] = pl_fadd(s[85][2], hinc);
            s[85][4] = pl_fadd(s[85][3], hinc);
        }
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:455-702.
__device__ void gsl_makeplume(float s[][202], int kmt, float& ztopmax, int ixx, int imm, int mpiid, float alpha) {
    char varn = 0;
    int izprint = 0;
    int iconv = 0;
    int itime = 0;
    int k = 0;
    int kk = 0;
    int kkmax = 0;
    int deltak = 0;
    int ilastprint = 0;
    int nrectotal = 0;
    int i_micro = 0;
    int n_sub_step = 0;
    float wmax = 0;
    float rmaxtime = 0;
    float es = 0;
    float esat = 0;
    float heat = 0;
    float dt_save = 0;
    char cixx = 0;
    float delz_thresold = __uint_as_float(0x42c80000u);
    int dtknt = 0;
    s[108][0] = 0;
    s[60][0] = __uint_as_float(0x40000000u);
    s[59][0] = __uint_as_float(0x43fa0000u);
    nrectotal = 150;
    dtknt = 0;
    s[91][0] = 1;
    ztopmax = __uint_as_float(0x00000000u);
    s[79][0] = __uint_as_float(0x00000000u);
    s[89][0] = __uint_as_float(0x00000000u);
    s[88][0] = __uint_as_float(0x3f800000u);
    wmax = __uint_as_float(0x3f800000u);
    kkmax = 10;
    deltak = 20;
    ilastprint = 0;
    s[63][0] = 1;
    gsl_initial(s, kmt, alpha);
    izprint = 0;
    rmaxtime = float(int(s[93][0]));
    while ((s[89][0] <= rmaxtime)) {
        s[62][0] = min(kmt, (kkmax + deltak));
        s[88][0] = pl_fmax(__uint_as_float(0x3c23d70au), pl_fmin(__uint_as_float(0x40a00000u), pl_fdiv(pl_fsub(s[25][2], s[25][1]), pl_fmul(s[60][0], wmax))));
        dtknt = (dtknt + 1);
        s[89][0] = pl_fadd(s[89][0], s[88][0]);
        s[108][0] += 1;
        s[91][0] = (1 + (int(s[89][0]) / 60));
        wmax = __uint_as_float(0x3f800000u);
        gsl_tend0_plumerise(s);
        s[63][0] = 1;
        gsl_lbound(s, alpha);
        gsl_vel_advectc_plumerise(s, int(s[62][0]), s[15], s[16], s[9], s[23]);
        gsl_scl_advectc_plumerise(s, 's', int(s[62][0]));
        gsl_scl_misc(s, int(s[62][0]), alpha);
        gsl_scl_dyn_entrain(s, int(s[62][0]), 200, s[73][0], s[0], s[72][0], alpha, s[84], s[17], s[1], s[39], s[18], s[2], s[40], s[19], s[3], s[20], s[4], s[21], s[5], s[102], s[103], s[105], s[104], s[106]);
        gsl_damp_grav_wave(s, 1, int(s[62][0]), deltak, s[88][0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40]);
        dt_save = s[88][0];
        n_sub_step = 3;
        s[88][0] = pl_fdiv(s[88][0], float(n_sub_step));
        for (i_micro=1; i_micro<=n_sub_step; i_micro+=1) {
            gsl_fallpart(s, int(s[62][0]));
            s[63][0] = 2;
            while ((int(s[63][0]) <= (int(s[62][0]) - 1))) {
                s[73][0] = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][int(s[63][0])], s[0][(int(s[63][0]) - 1)]));
                es = gsl_esat_pr(s, s[1][int(s[63][0])]);
                s[12][int(s[63][0])] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][int(s[63][0])], es));
                s[11][int(s[63][0])] = es;
                s[9][int(s[63][0])] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][int(s[63][0])]), s[1][int(s[63][0])]);
                if ((s[0][int(s[63][0])] >= __uint_as_float(0x00000000u))) {
                    s[57][0] = pl_fdiv(pl_fsub(s[12][(int(s[63][0]) + 1)], s[12][(int(s[63][0]) - 1)]), pl_fsub(s[26][(int(s[63][0]) + 1)], s[26][(int(s[63][0]) - 1)]));
                } else {
                    s[57][0] = pl_fdiv(pl_fsub(s[12][(int(s[63][0]) + 1)], s[12][(int(s[63][0]) - 1)]), pl_fsub(s[26][(int(s[63][0]) + 1)], s[26][(int(s[63][0]) - 1)]));
                }
                gsl_waterbal(s);
                s[63][0] = (int(s[63][0]) + 1);
            }
        }
        s[88][0] = dt_save;
        gsl_visc_w(s, int(s[62][0]), deltak, kmt);
        gsl_update_plumerise(s, int(s[62][0]), 's');
        gsl_hadvance_plumerise(s, 1, int(s[62][0]), s[88][0], s[15], s[16], s[0], int(s[91][0]));
        gsl_buoyancy_plumerise(s, int(s[62][0]), s[1], s[39], s[2], s[40], s[4], s[5], s[3], s[16], s[33]);
        gsl_entrainment(s, int(s[62][0]), s[0], s[16], s[84], alpha);
        gsl_update_plumerise(s, int(s[62][0]), 'w');
        gsl_hadvance_plumerise(s, 2, int(s[62][0]), s[88][0], s[15], s[16], s[0], int(s[91][0]));
        for (k=2; k<=int(s[62][0]); k+=1) {
            es = gsl_esat_pr(s, s[1][k]);
            s[12][k] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][k], es));
            s[11][k] = es;
            s[10][k] = pl_fsub(s[1][k], s[39][k]);
            s[9][k] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][k]), s[1][k]);
            if ((pl_fabs(s[15][k]) > wmax)) {
                wmax = pl_fabs(s[15][k]);
            }
        }
        gsl_damp_grav_wave(s, 2, int(s[62][0]), deltak, s[88][0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40]);
        for (k=2; k<=int(s[62][0]); k+=1) {
            s[84][k] = s[104][k];
        }
        kk = 1;
        while ((s[0][kk] > __uint_as_float(0x3f800000u))) {
            kk = (kk + 1);
            s[79][0] = s[25][kk];
        }
        s[107][int(s[91][0])] = s[79][0];
        ztopmax = pl_fmax(s[79][0], ztopmax);
        kkmax = max(kk, kkmax);
        if ((int(s[91][0]) > 10)) {
            if ((pl_fabs(pl_fsub(s[107][int(s[91][0])], s[107][(int(s[91][0]) - 10)])) < delz_thresold)) {
                break;
            }
        }
    }
    if (imm==1) {
        s[111][0] = ztopmax;
    } else {
        s[112][0] = ztopmax;
    }
    if (imm==1) {
        s[109][0] = s[108][0];
    } else {
        s[110][0] = s[108][0];
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:710-738.
__device__ void gsl_burn(float s[][202], float& eflux, float& water) {
    if ((s[89][0] > s[90][0])) {
        eflux = __uint_as_float(0x358637bdu);
        water = __uint_as_float(0x00000000u);
        return;
    } else {
        eflux = s[85][int(s[91][0])];
        water = pl_fdiv(pl_fmul(pl_fmul(eflux, pl_fdiv(s[88][0], __uint_as_float(0x4b933f50u))), pl_fadd(__uint_as_float(0x3f000000u), s[86][0])), __uint_as_float(0x3f0ccccdu));
        water = pl_fmul(water, __uint_as_float(0x447a0000u));
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:758-842.
__device__ void gsl_lbound(float s[][202], float alpha) {
    float es = 0;
    float esat = 0;
    float eflux = 0;
    float water = 0;
    float pres = 0;
    float c1 = 0;
    float c2 = 0;
    float f = 0;
    float zv = 0;
    float denscor = 0;
    float xwater = 0;
    s[4][1] = s[4][2];
    s[5][1] = s[5][2];
    s[3][1] = __uint_as_float(0x00000000u);
    gsl_burn(s, eflux, water);
    pres = pl_fmul(s[38][1], __uint_as_float(0x447a0000u));
    c1 = pl_fdiv(__uint_as_float(0x40a00000u), pl_fmul(__uint_as_float(0x40c00000u), alpha));
    c2 = pl_fmul(__uint_as_float(0x3f666666u), alpha);
    f = pl_fdiv(eflux, pl_fmul(pl_fmul(pres, __uint_as_float(0x447b2666u)), __uint_as_float(0x40490fd0u)));
    f = pl_fmul(pl_fmul(__uint_as_float(0x452ff46eu), f), s[81][0]);
    zv = pl_fmul(c1, s[82][0]);
    s[0][1] = pl_fdiv(pl_fmul(c1, pl_fpow(pl_fmul(c2, f), __uint_as_float(0x3eaaaaabu))), pl_fpow(zv, __uint_as_float(0x3eaaaaabu)));
    denscor = pl_fdiv(pl_fdiv(pl_fdiv(pl_fmul(c1, f), __uint_as_float(0x411ced67u)), pl_fpow(pl_fmul(c2, f), __uint_as_float(0x3eaaaaabu))), pl_fpow(zv, __uint_as_float(0x3fd55555u)));
    s[1][1] = pl_fdiv(s[39][1], pl_fsub(__uint_as_float(0x3f800000u), denscor));
    s[15][1] = s[0][1];
    s[103][1] = __uint_as_float(0x00000000u);
    s[104][1] = s[82][0];
    s[7][1] = __uint_as_float(0xc0800000u);
    s[8][1] = __uint_as_float(0xc0400000u);
    s[10][1] = pl_fsub(s[1][1], s[39][1]);
    s[58][1] = s[59][0];
    s[9][1] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][1]), s[1][1]);
    xwater = pl_fdiv(water, pl_fmax(__uint_as_float(0x1e3ce508u), pl_fmul(pl_fmul(s[0][1], s[88][0]), s[9][1])));
    s[2][1] = pl_fadd(xwater, s[40][1]);
    es = gsl_esat_pr(s, s[1][1]);
    s[11][1] = es;
    s[12][1] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fmax(__uint_as_float(0x1e3ce508u), pl_fsub(s[38][1], es)));
    if ((s[2][1] > s[12][1])) {
        s[3][1] = pl_fadd(pl_fsub(s[2][1], s[12][1]), s[3][1]);
        s[2][1] = s[12][1];
    }
    gsl_waterbal(s);
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:850-913.
__device__ void gsl_initial(float s[][202], int kmt, float alpha) {
    int isub = 0;
    int k = 0;
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int lbuoy = 0;
    int itmp = 0;
    int isubm1 = 0;
    float xn1 = 0;
    float xi = 0;
    float es = 0;
    float esat = 0;
    s[61][0] = kmt;
    for (k=1; k<=int(s[61][0]); k+=1) {
        s[10][k] = __uint_as_float(0x00000000u);
        s[0][k] = __uint_as_float(0x00000000u);
        s[1][k] = s[39][k];
        s[15][k] = __uint_as_float(0x00000000u);
        s[16][k] = __uint_as_float(0x00000000u);
        s[2][k] = s[40][k];
        s[7][k] = __uint_as_float(0x00000000u);
        s[8][k] = __uint_as_float(0x00000000u);
        s[4][k] = __uint_as_float(0x00000000u);
        s[5][k] = __uint_as_float(0x00000000u);
        s[3][k] = __uint_as_float(0x00000000u);
        es = gsl_esat_pr(s, s[1][k]);
        s[11][k] = es;
        s[12][k] = pl_fdiv(pl_fmul(__uint_as_float(0x3f1f3b64u), es), pl_fsub(s[38][k], es));
        s[9][k] = pl_fdiv(pl_fmul(__uint_as_float(0x4559bccdu), s[38][k]), s[1][k]);
        s[103][k] = __uint_as_float(0x00000000u);
        s[104][k] = __uint_as_float(0x00000000u);
    }
    s[84][1] = s[82][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[84][k] = pl_fadd(s[84][(k - 1)], pl_fmul(pl_fmul(__uint_as_float(0x3f99999au), alpha), pl_fsub(s[26][k], s[26][(k - 1)])));
    }
    s[84][1] = s[82][0];
    s[104][1] = s[82][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[84][k] = pl_fadd(s[84][(k - 1)], pl_fmul(pl_fmul(__uint_as_float(0x3f99999au), alpha), pl_fsub(s[26][k], s[26][(k - 1)])));
        s[104][k] = s[84][k];
    }
    s[58][1] = s[59][0];
    for (k=2; k<=int(s[61][0]); k+=1) {
        s[58][k] = pl_fmax(__uint_as_float(0x3a83126fu), pl_fsub(s[58][(k - 1)], pl_fdiv(pl_fmul(__uint_as_float(0x3f800000u), s[59][0]), __uint_as_float(0x43480000u))));
    }
    gsl_lbound(s, alpha);
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:918-934.
__device__ void gsl_damp_grav_wave(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* w, float* t, float* tt, float* qv, float* qh, float* qi, float* qc, float* te, float* pe, float* qvenv) {
    float* dummy = s[127];  // workspace row 127: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dummy[_z] = 0.0f;
    int _a = 0;
    if ((ifrom == 1)) {
        gsl_friction(s, ifrom, nm1, deltak, dt, zt, zm, t, tt, te);
        return;
    }
    for (int _a=1; _a<=200; ++_a) dummy[_a] = __uint_as_float(0x00000000u);
    if ((ifrom == 2)) {
        gsl_friction(s, ifrom, nm1, deltak, dt, zt, zm, w, dummy, dummy);
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:939-969.
__device__ void gsl_friction(float s[][202], int ifrom, int nm1, int deltak, float dt, float* zt, float* zm, float* var1, float* vart, float* var2) {
    int k = 0;
    int nfpt = 0;
    int kf = 0;
    float zmkf = 0;
    float ztop = 0;
    float distim = 0;
    float c1 = 0;
    float c2 = 0;
    kf = (nm1 - int(deltak));
    zmkf = zm[kf];
    ztop = zm[nm1];
    distim = pl_fmin(pl_fmul(__uint_as_float(0x40400000u), dt), __uint_as_float(0x42700000u));
    c1 = pl_fdiv(__uint_as_float(0x3f800000u), pl_fmul(distim, pl_fsub(ztop, zmkf)));
    c2 = pl_fmul(dt, c1);
    if ((ifrom == 1)) {
        for (k=nm1; k>=2; k+=-1) {
            if ((zt[k] <= zmkf)) {
                continue;
            }
            vart[k] = pl_fadd(vart[k], pl_fmul(pl_fmul(c1, pl_fsub(zt[k], zmkf)), pl_fsub(var2[k], var1[k])));
        }
    } else if ((ifrom == 2)) {
        for (k=nm1; k>=2; k+=-1) {
            if ((zt[k] <= zmkf)) {
                continue;
            }
            var1[k] = pl_fadd(var1[k], pl_fmul(pl_fmul(c2, pl_fsub(zt[k], zmkf)), pl_fsub(var2[k], var1[k])));
        }
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:975-1005.
__device__ void gsl_vel_advectc_plumerise(float s[][202], int m1, float* wc, float* wt, float* rho, float* dzm) {
    int k = 0;
    float* flxw = s[128];  // workspace row 128: was a 202-word local array
    for (int _z=0; _z<202; ++_z) flxw[_z] = 0.0f;
    float* dn0 = s[129];  // workspace row 129: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dn0[_z] = 0.0f;
    float c1z = 0;
    int _a = 0;
    for (int _a=1; _a<=m1; ++_a) dn0[_a] = pl_fmul(rho[_a], __uint_as_float(0x3a83126fu));
    flxw[1] = pl_fmul(wc[1], dn0[1]);
    for (k=2; k<=(m1 - 1); k+=1) {
        flxw[k] = pl_fmul(pl_fmul(wc[k], __uint_as_float(0x3f000000u)), pl_fadd(dn0[k], dn0[(k + 1)]));
    }
    c1z = __uint_as_float(0x3f000000u);
    for (k=2; k<=(m1 - 2); k+=1) {
        wt[k] = pl_fadd(wt[k], pl_fmul(pl_fdiv(pl_fmul(c1z, dzm[k]), pl_fadd(dn0[k], dn0[(k + 1)])), pl_fadd(pl_fsub(pl_fmul(pl_fadd(flxw[k], flxw[(k - 1)]), pl_fadd(wc[k], wc[(k - 1)])), pl_fmul(pl_fadd(flxw[k], flxw[(k + 1)]), pl_fadd(wc[k], wc[(k + 1)]))), pl_fmul(pl_fmul(pl_fsub(flxw[(k + 1)], flxw[(k - 1)]), __uint_as_float(0x40000000u)), wc[k]))));
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1011-1031.
__device__ void gsl_hadvance_plumerise(float s[][202], int iac, int m1, float dt, float* wc, float* wt, float* wp, int mintime) {
    int k = 0;
    float* dummy = s[130];  // workspace row 130: was a 202-word local array
    for (int _z=0; _z<202; ++_z) dummy[_z] = 0.0f;
    float eps = 0;
    eps = __uint_as_float(0x3e4ccccdu);
    if ((mintime == 1)) {
        eps = __uint_as_float(0x3f000000u);
    }
    gsl_predict_plumerise(s, m1, wc, wp, wt, dummy, iac, pl_fmul(__uint_as_float(0x40000000u), dt), eps);
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1036-1077.
__device__ void gsl_predict_plumerise(float s[][202], int npts, float* ac, float* ap, float* fa, float* af, int iac, float dtlp, float epsu) {
    int m = 0;
    if ((iac == 1)) {
        for (m=1; m<=npts; m+=1) {
            ac[m] = pl_fadd(ac[m], pl_fmul(epsu, pl_fsub(ap[m], pl_fmul(__uint_as_float(0x40000000u), ac[m]))));
        }
        return;
    } else if ((iac == 2)) {
        for (m=1; m<=npts; m+=1) {
            af[m] = ap[m];
            ap[m] = pl_fadd(ac[m], pl_fmul(epsu, af[m]));
        }
    }
    for (m=1; m<=npts; m+=1) {
        ac[m] = af[m];
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1082-1112.
__device__ void gsl_buoyancy_plumerise(float s[][202], int m1, float* t, float* te, float* qv, float* qvenv, float* qh, float* qi, float* qc, float* wt, float* scr1) {
    int k = 0;
    float tv = 0;
    float tve = 0;
    float qwtotl = 0;
    float umgamai = 0;
    umgamai = __uint_as_float(0x3f2aaaabu);
    for (k=2; k<=(m1 - 1); k+=1) {
        tv = pl_fdiv(pl_fmul(t[k], pl_fadd(__uint_as_float(0x3f800000u), pl_fdiv(qv[k], __uint_as_float(0x3f1f3b64u)))), pl_fadd(__uint_as_float(0x3f800000u), qv[k]));
        tve = pl_fdiv(pl_fmul(te[k], pl_fadd(__uint_as_float(0x3f800000u), pl_fdiv(qvenv[k], __uint_as_float(0x3f1f3b64u)))), pl_fadd(__uint_as_float(0x3f800000u), qvenv[k]));
        qwtotl = pl_fadd(pl_fadd(qh[k], qi[k]), qc[k]);
        scr1[k] = pl_fmul(pl_fmul(__uint_as_float(0x411ccccdu), umgamai), pl_fsub(pl_fdiv(pl_fsub(tv, tve), tve), qwtotl));
    }
    for (k=2; k<=(m1 - 2); k+=1) {
        wt[k] = pl_fadd(wt[k], pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(scr1[k], scr1[(k + 1)])));
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1118-1158.
__device__ void gsl_entrainment(float s[][202], int m1, float* w, float* wt, float* radius, float alpha) {
    int k = 0;
    float dmdtm = 0;
    float wbar = 0;
    float radius_bar = 0;
    float umgamai = 0;
    float dyn_entr = 0;
    umgamai = __uint_as_float(0x3f2aaaabu);
    for (k=2; k<=(m1 - 1); k+=1) {
        wbar = w[k];
        radius_bar = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(radius[k], radius[(k - 1)]));
        dmdtm = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(umgamai, __uint_as_float(0x40000000u)), alpha), pl_fabs(wbar)), radius_bar);
        wt[k] = pl_fsub(wt[k], pl_fmul(dmdtm, pl_fabs(wbar)));
        dyn_entr = pl_fdiv(pl_fmul(__uint_as_float(0x3ea2f96bu), pl_fabs(pl_fsub(pl_fadd(pl_fsub(s[103][k], s[102][k]), s[103][(k - 1)]), s[102][(k - 1)]))), radius_bar);
        wt[k] = pl_fsub(wt[k], pl_fmul(dyn_entr, pl_fabs(wbar)));
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1164-1287.
__device__ void gsl_scl_advectc_plumerise(float s[][202], char varn, int mzp) {
    float dtlto2 = 0;
    int k = 0;
    int _a = 0;
    dtlto2 = pl_fmul(__uint_as_float(0x3f000000u), s[88][0]);
    s[29][1] = pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][1], s[15][1]), dtlto2), s[9][1]), __uint_as_float(0x3a83126fu));
    s[30][1] = pl_fmul(pl_fmul(pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][1], s[15][1])), dtlto2), s[23][1]);
    for (k=2; k<=mzp; k+=1) {
        s[29][k] = pl_fmul(pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][k], s[15][k]), dtlto2), __uint_as_float(0x3f000000u)), pl_fadd(s[9][k], s[9][(k + 1)])), __uint_as_float(0x3a83126fu));
        s[30][k] = pl_fmul(pl_fmul(pl_fmul(pl_fadd(s[0][k], s[15][k]), dtlto2), __uint_as_float(0x3f000000u)), s[23][k]);
    }
    for (k=1; k<=mzp; k+=1) {
        s[27][k] = pl_fmul(pl_fsub(s[26][(k + 1)], s[25][k]), s[23][k]);
        s[28][k] = pl_fmul(pl_fsub(s[25][k], s[26][k]), s[23][k]);
        s[31][k] = pl_fdiv(s[24][k], pl_fmul(s[9][k], __uint_as_float(0x3a83126fu)));
    }
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[1][_a];
    gsl_fa_zc_plumerise(s, mzp, s[1], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[1], s[33], s[17], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[2][_a];
    gsl_fa_zc_plumerise(s, mzp, s[2], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[2], s[33], s[18], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[3][_a];
    gsl_fa_zc_plumerise(s, mzp, s[3], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[3], s[33], s[19], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[5][_a];
    gsl_fa_zc_plumerise(s, mzp, s[5], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[5], s[33], s[21], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[4][_a];
    gsl_fa_zc_plumerise(s, mzp, s[4], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[4], s[33], s[20], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[103][_a];
    gsl_fa_zc_plumerise(s, mzp, s[103], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[103], s[33], s[105], s[88][0]);
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[104][_a];
    gsl_fa_zc_plumerise(s, mzp, s[104], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[104], s[33], s[106], s[88][0]);
    return;
    for (int _a=1; _a<=200; ++_a) s[33][_a] = s[6][_a];
    gsl_fa_zc_plumerise(s, mzp, s[6], s[33], s[29], s[30], s[32], s[31], s[27], s[28]);
    gsl_advtndc_plumerise(s, mzp, s[6], s[33], s[22], s[88][0]);
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1293-1334.
__device__ void gsl_fa_zc_plumerise(float s[][202], int m1, float* scp, float* scr1, float* vt3dc, float* vt3df, float* vt3dg, float* vt3dk, float* vctr1, float* vctr2) {
    int k = 0;
    float dfact = 0;
    dfact = __uint_as_float(0x3f000000u);
    for (k=1; k<=(m1 - 1); k+=1) {
        vt3dg[k] = pl_fmul(vt3dc[k], pl_fadd(pl_fadd(pl_fmul(vctr1[k], scr1[k]), pl_fmul(vctr2[k], scr1[(k + 1)])), pl_fmul(vt3df[k], pl_fsub(scr1[k], scr1[(k + 1)]))));
    }
    for (k=1; k<=(m1 - 1); k+=1) {
        if ((vt3dc[k] > __uint_as_float(0x00000000u))) {
            if ((pl_fmul(vt3dg[k], vt3dk[k]) > pl_fmul(dfact, scr1[k]))) {
                vt3dg[k] = pl_fmul(vt3dc[k], scr1[k]);
            }
        } else if ((vt3dc[k] < __uint_as_float(0x00000000u))) {
            if ((pl_fmul((-vt3dg[k]), vt3dk[(k + 1)]) > pl_fmul(dfact, scr1[(k + 1)]))) {
                vt3dg[k] = pl_fmul(vt3dc[k], scr1[(k + 1)]);
            }
        }
    }
    for (k=2; k<=(m1 - 1); k+=1) {
        scr1[k] = pl_fadd(scr1[k], pl_fmul(vt3dk[k], pl_fadd(pl_fsub(vt3dg[(k - 1)], vt3dg[k]), pl_fmul(scp[k], pl_fsub(vt3dc[k], vt3dc[(k - 1)])))));
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1339-1348.
__device__ void gsl_advtndc_plumerise(float s[][202], int m1, float* scp, float* sca, float* sct, float dtl) {
    int k = 0;
    float dtli = 0;
    dtli = pl_fdiv(__uint_as_float(0x3f800000u), dtl);
    for (k=2; k<=(m1 - 1); k+=1) {
        sct[k] = pl_fadd(sct[k], pl_fmul(pl_fsub(sca[k], scp[k]), dtli));
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1353-1362.
__device__ void gsl_tend0_plumerise(float s[][202]) {
    int _a = 0;
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[16][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[17][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[18][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[19][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[20][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[21][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[105][_a] = __uint_as_float(0x00000000u);
    for (int _a=1; _a<=int(s[62][0]); ++_a) s[106][_a] = __uint_as_float(0x00000000u);
}

// physics/smoke_dust/module_smoke_plumerise.F90:1370-1402.
__device__ void gsl_scl_misc(float s[][202], int m1, float alpha) {
    int k = 0;
    float dmdtm = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        s[73][0] = pl_fmul(__uint_as_float(0x3f000000u), pl_fadd(s[0][k], s[0][(k - 1)]));
        s[72][0] = pl_fdiv(pl_fmul((-s[73][0]), __uint_as_float(0x411cf5c3u)), __uint_as_float(0x447b0000u));
        dmdtm = pl_fdiv(pl_fmul(pl_fmul(__uint_as_float(0x40000000u), alpha), pl_fabs(s[73][0])), s[84][k]);
        s[17][k] = pl_fsub(pl_fadd(s[17][k], s[72][0]), pl_fmul(dmdtm, pl_fsub(s[1][k], s[39][k])));
        s[18][k] = pl_fsub(s[18][k], pl_fmul(dmdtm, pl_fsub(s[2][k], s[40][k])));
        s[19][k] = pl_fsub(s[19][k], pl_fmul(dmdtm, s[3][k]));
        s[20][k] = pl_fsub(s[20][k], pl_fmul(dmdtm, s[4][k]));
        s[21][k] = pl_fsub(s[21][k], pl_fmul(dmdtm, s[5][k]));
        s[105][k] = pl_fsub(s[105][k], pl_fmul(dmdtm, pl_fsub(s[103][k], s[102][k])));
        s[106][k] = pl_fadd(s[106][k], pl_fmul(pl_fmul(pl_fmul(__uint_as_float(0x3f000000u), dmdtm), __uint_as_float(0x3f99999au)), s[84][k]));
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1408-1471.
__device__ void gsl_scl_dyn_entrain(float s[][202], int m1, int nkp, float wbar, float* w, float adiabat, float alpha, float* radius, float* tt, float* t, float* te, float* qvt, float* qv, float* qvenv, float* qct, float* qc, float* qht, float* qh, float* qit, float* qi, float* vel_e, float* vel_p, float* vel_t, float* rad_p, float* rad_t) {
    int k = 0;
    float dmdtm = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        rad_t[k] = pl_fadd(rad_t[k], pl_fdiv(pl_fabs(pl_fsub(vel_e[k], vel_p[k])), __uint_as_float(0x40490ff9u)));
        dmdtm = pl_fdiv(pl_fmul(__uint_as_float(0x3f22f96bu), pl_fabs(pl_fsub(vel_e[k], vel_p[k]))), radius[k]);
        vel_t[k] = pl_fsub(vel_t[k], pl_fmul(dmdtm, pl_fsub(vel_p[k], vel_e[k])));
        tt[k] = pl_fsub(tt[k], pl_fmul(dmdtm, pl_fsub(t[k], te[k])));
        qvt[k] = pl_fsub(qvt[k], pl_fmul(dmdtm, pl_fsub(qv[k], qvenv[k])));
        qct[k] = pl_fsub(qct[k], pl_fmul(dmdtm, qc[k]));
        qht[k] = pl_fsub(qht[k], pl_fmul(dmdtm, qh[k]));
        qit[k] = pl_fsub(qit[k], pl_fmul(dmdtm, qi[k]));
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1478-1525.
__device__ void gsl_visc_w(float s[][202], int m1, int deltak, int kmt) {
    int k = 0;
    int m2 = 0;
    float dz1t = 0;
    float dz1m = 0;
    float dz2t = 0;
    float dz2m = 0;
    float d2wdz = 0;
    float d2tdz = 0;
    float d2qvdz = 0;
    float d2qhdz = 0;
    float d2qcdz = 0;
    float d2qidz = 0;
    float d2scdz = 0;
    float d2vel_pdz = 0;
    float d2rad_dz = 0;
    bool printed = false;
    m2 = min(m1, kmt);
    for (k=2; k<=(m2 - 1); k+=1) {
        dz1t = pl_fmul(__uint_as_float(0x3f000000u), pl_fsub(s[26][(k + 1)], s[26][(k - 1)]));
        dz2t = pl_fdiv(s[58][k], pl_fmul(dz1t, dz1t));
        dz1m = pl_fmul(__uint_as_float(0x3f000000u), pl_fsub(s[25][(k + 1)], s[25][(k - 1)]));
        dz2m = pl_fdiv(s[58][k], pl_fmul(dz1m, dz1m));
        d2wdz = pl_fmul(pl_fadd(pl_fsub(s[0][(k + 1)], pl_fmul(2, s[0][k])), s[0][(k - 1)]), dz2m);
        d2tdz = pl_fmul(pl_fadd(pl_fsub(s[1][(k + 1)], pl_fmul(2, s[1][k])), s[1][(k - 1)]), dz2t);
        d2qvdz = pl_fmul(pl_fadd(pl_fsub(s[2][(k + 1)], pl_fmul(2, s[2][k])), s[2][(k - 1)]), dz2t);
        d2qhdz = pl_fmul(pl_fadd(pl_fsub(s[4][(k + 1)], pl_fmul(2, s[4][k])), s[4][(k - 1)]), dz2t);
        d2qcdz = pl_fmul(pl_fadd(pl_fsub(s[3][(k + 1)], pl_fmul(2, s[3][k])), s[3][(k - 1)]), dz2t);
        d2qidz = pl_fmul(pl_fadd(pl_fsub(s[5][(k + 1)], pl_fmul(2, s[5][k])), s[5][(k - 1)]), dz2t);
        d2vel_pdz = pl_fmul(pl_fadd(pl_fsub(s[103][(k + 1)], pl_fmul(2, s[103][k])), s[103][(k - 1)]), dz2t);
        d2rad_dz = pl_fmul(pl_fadd(pl_fsub(s[104][(k + 1)], pl_fmul(2, s[104][k])), s[104][(k - 1)]), dz2t);
        s[16][k] = pl_fadd(s[16][k], d2wdz);
        s[17][k] = pl_fadd(s[17][k], d2tdz);
        s[18][k] = pl_fadd(s[18][k], d2qvdz);
        s[19][k] = pl_fadd(s[19][k], d2qcdz);
        s[20][k] = pl_fadd(s[20][k], d2qhdz);
        s[21][k] = pl_fadd(s[21][k], d2qidz);
        s[105][k] = pl_fadd(s[105][k], d2vel_pdz);
        s[106][k] = pl_fadd(s[106][k], d2rad_dz);
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1533-1573.
__device__ void gsl_update_plumerise(float s[][202], int m1, char varn) {
    int k = 0;
    if ((varn == 'w')) {
        for (k=2; k<=(m1 - 1); k+=1) {
            s[0][k] = pl_fadd(s[0][k], pl_fmul(s[16][k], s[88][0]));
        }
        return;
    } else {
        for (k=2; k<=(m1 - 1); k+=1) {
            s[1][k] = pl_fadd(s[1][k], pl_fmul(s[17][k], s[88][0]));
            s[2][k] = pl_fadd(s[2][k], pl_fmul(s[18][k], s[88][0]));
            s[3][k] = pl_fadd(s[3][k], pl_fmul(s[19][k], s[88][0]));
            s[4][k] = pl_fadd(s[4][k], pl_fmul(s[20][k], s[88][0]));
            s[5][k] = pl_fadd(s[5][k], pl_fmul(s[21][k], s[88][0]));
            s[2][k] = pl_fmax(__uint_as_float(0x00000000u), s[2][k]);
            s[3][k] = pl_fmax(__uint_as_float(0x00000000u), s[3][k]);
            s[4][k] = pl_fmax(__uint_as_float(0x00000000u), s[4][k]);
            s[5][k] = pl_fmax(__uint_as_float(0x00000000u), s[5][k]);
            s[103][k] = pl_fadd(s[103][k], pl_fmul(s[105][k], s[88][0]));
            s[104][k] = pl_fadd(s[104][k], pl_fmul(s[106][k], s[88][0]));
        }
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1579-1639.
__device__ void gsl_fallpart(float s[][202], int m1) {
    int k = 0;
    float vtc = 0;
    float dfhz = 0;
    float dfiz = 0;
    float dz1 = 0;
    for (k=2; k<=(m1 - 1); k+=1) {
        vtc = pl_fmul(__uint_as_float(0x40a36fb7u), pl_fpow(s[9][k], __uint_as_float(0x3e000000u)));
        s[7][k] = __uint_as_float(0xc0800000u);
        s[75][0] = pl_fadd(s[0][k], s[7][k]);
        s[70][k] = pl_fadd(__uint_as_float(0x3fcccccdu), pl_fmul(__uint_as_float(0x3a156c0du), pl_fpow(pl_fabs(s[75][0]), __uint_as_float(0x3fc00000u))));
        s[8][k] = __uint_as_float(0xc0400000u);
        s[76][0] = pl_fadd(s[0][k], s[8][k]);
        s[71][k] = pl_fadd(__uint_as_float(0x3fcccccdu), pl_fdiv(pl_fmul(__uint_as_float(0x3a156c0du), pl_fpow(pl_fabs(s[76][0]), __uint_as_float(0x3fc00000u))), __uint_as_float(0x3f400000u)));
        if ((s[75][0] >= __uint_as_float(0x00000000u))) {
            dfhz = pl_fdiv(pl_fmul(s[4][k], pl_fsub(pl_fmul(s[9][k], s[7][k]), pl_fmul(s[9][(k - 1)], s[7][(k - 1)]))), s[9][(k - 1)]);
        } else {
            dfhz = pl_fdiv(pl_fmul(s[4][k], pl_fsub(pl_fmul(s[9][(k + 1)], s[7][(k + 1)]), pl_fmul(s[9][k], s[7][k]))), s[9][k]);
        }
        if ((s[76][0] >= __uint_as_float(0x00000000u))) {
            dfiz = pl_fdiv(pl_fmul(s[5][k], pl_fsub(pl_fmul(s[9][k], s[8][k]), pl_fmul(s[9][(k - 1)], s[8][(k - 1)]))), s[9][(k - 1)]);
        } else {
            dfiz = pl_fdiv(pl_fmul(s[5][k], pl_fsub(pl_fmul(s[9][(k + 1)], s[8][(k + 1)]), pl_fmul(s[9][k], s[8][k]))), s[9][k]);
        }
        dz1 = pl_fsub(s[25][k], s[25][(k - 1)]);
        s[20][k] = pl_fsub(s[20][k], pl_fdiv(dfhz, dz1));
        s[21][k] = pl_fsub(s[21][k], pl_fdiv(dfiz, dz1));
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:1644-1667.
__device__ void gsl_waterbal(float s[][202]) {
    if ((s[3][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    if ((s[4][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    if ((s[5][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
        s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    gsl_evaporate(s);
    gsl_sublimate(s);
    gsl_glaciate(s);
    gsl_melt(s);
    gsl_convert(s);
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1675-1881.
__device__ void gsl_evaporate(float s[][202]) {
    float evhdt = 0;
    float evidt = 0;
    float evrate = 0;
    float evap = 0;
    float sd = 0;
    float quant = 0;
    float dividend = 0;
    float divisor = 0;
    float devidt = 0;
    sd = pl_fsub(s[12][int(s[63][0])], s[2][int(s[63][0])]);
    if ((sd == __uint_as_float(0x00000000u))) {
        return;
    }
    evhdt = __uint_as_float(0x00000000u);
    evidt = __uint_as_float(0x00000000u);
    evrate = pl_fabs(pl_fmul(s[73][0], s[57][0]));
    evap = pl_fmul(evrate, s[88][0]);
    if ((sd <= __uint_as_float(0x00000000u))) {
        if ((evap >= pl_fabs(sd))) {
            s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
            s[2][int(s[63][0])] = s[12][int(s[63][0])];
            s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
            return;
        } else {
            s[3][int(s[63][0])] = pl_fadd(s[3][int(s[63][0])], evap);
            s[2][int(s[63][0])] = pl_fsub(s[2][int(s[63][0])], evap);
            s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(evap, __uint_as_float(0x451ba0a4u)));
            return;
        }
    } else {
        if ((evap <= s[3][int(s[63][0])])) {
            if ((sd <= evap)) {
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                return;
            } else {
                sd = pl_fsub(sd, evap);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evap);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evap, __uint_as_float(0x451ba0a4u)));
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], evap);
            }
        } else {
            if ((sd <= s[3][int(s[63][0])])) {
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], sd);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                return;
            } else {
                sd = pl_fsub(sd, s[3][int(s[63][0])]);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[3][int(s[63][0])]);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[3][int(s[63][0])], __uint_as_float(0x451ba0a4u)));
                s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
            }
        }
        if ((s[4][int(s[63][0])] > __uint_as_float(0x2edbe6ffu))) {
            quant = pl_fmul(pl_fsub(pl_fsub(s[12][int(s[63][0])], s[3][int(s[63][0])]), s[2][int(s[63][0])]), s[9][int(s[63][0])]);
            evhdt = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(s[88][0], __uint_as_float(0x3a0e9b39u)), quant), pl_fpow(pl_fmul(s[4][int(s[63][0])], s[9][int(s[63][0])]), __uint_as_float(0x3f266666u))), s[9][int(s[63][0])]);
            if ((evhdt <= s[4][int(s[63][0])])) {
                if ((sd <= evhdt)) {
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], sd);
                    s[2][int(s[63][0])] = s[12][int(s[63][0])];
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                    return;
                } else {
                    sd = pl_fsub(sd, evhdt);
                    s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evhdt);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evhdt, __uint_as_float(0x451ba0a4u)));
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], evhdt);
                }
            } else {
                if ((sd <= s[4][int(s[63][0])])) {
                    s[2][int(s[63][0])] = s[12][int(s[63][0])];
                    s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], sd);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x451ba0a4u)));
                    return;
                } else {
                    sd = pl_fsub(sd, s[4][int(s[63][0])]);
                    s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[4][int(s[63][0])]);
                    s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[4][int(s[63][0])], __uint_as_float(0x451ba0a4u)));
                    s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
                }
            }
        }
        if ((s[5][int(s[63][0])] <= __uint_as_float(0x2edbe6ffu))) {
            return;
        }
        dividend = pl_fmul(pl_fmul(pl_fmul(pl_fpow(pl_fdiv(__uint_as_float(0x49742400u), s[9][int(s[63][0])]), __uint_as_float(0x3ef33333u)), pl_fsub(pl_fdiv(sd, s[12][int(s[63][0])]), 1)), pl_fpow(s[5][int(s[63][0])], __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90a3d7u));
        divisor = pl_fadd(__uint_as_float(0x492ae600u), pl_fdiv(__uint_as_float(0x4a7a3e80u), pl_fmul(__uint_as_float(0x41200000u), s[11][int(s[63][0])])));
        devidt = pl_fdiv(pl_fmul((-s[71][int(s[63][0])]), dividend), divisor);
        evidt = pl_fmul(devidt, s[88][0]);
        if ((evidt <= s[5][int(s[63][0])])) {
            if ((sd <= evidt)) {
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], sd);
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x45306b59u)));
                return;
            } else {
                sd = pl_fsub(sd, evidt);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], evidt);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(evidt, __uint_as_float(0x45306b59u)));
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], evidt);
            }
        } else {
            if ((sd <= s[5][int(s[63][0])])) {
                s[2][int(s[63][0])] = s[12][int(s[63][0])];
                s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], sd);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(sd, __uint_as_float(0x45306b59u)));
                return;
            } else {
                sd = pl_fsub(sd, s[5][int(s[63][0])]);
                s[2][int(s[63][0])] = pl_fadd(s[2][int(s[63][0])], s[5][int(s[63][0])]);
                s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(s[5][int(s[63][0])], __uint_as_float(0x45306b59u)));
                s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
            }
        }
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1890-1958.
__device__ void gsl_convert(float s[][202]) {
    float accrete = 0;
    float con = 0;
    float q = 0;
    float h = 0;
    float bc1 = 0;
    float bc2 = 0;
    float total = 0;
    if ((s[1][int(s[63][0])] <= __uint_as_float(0x4386a666u))) {
        return;
    }
    if ((s[3][int(s[63][0])] == __uint_as_float(0x00000000u))) {
        return;
    }
    accrete = __uint_as_float(0x00000000u);
    con = __uint_as_float(0x00000000u);
    q = pl_fmul(s[9][int(s[63][0])], s[3][int(s[63][0])]);
    h = pl_fmul(s[9][int(s[63][0])], s[4][int(s[63][0])]);
    if ((s[4][int(s[63][0])] > __uint_as_float(0x00000000u))) {
        accrete = pl_fmul(pl_fmul(__uint_as_float(0x3baa64c3u), q), pl_fpow(h, __uint_as_float(0x3f600000u)));
    }
    if ((1 != 0)) {
        con = pl_fdiv(pl_fmul(pl_fmul(pl_fmul(q, q), q), __uint_as_float(0x3e158106u)), pl_fmul(__uint_as_float(0x42700000u), pl_fadd(pl_fmul(pl_fmul(__uint_as_float(0x40a00000u), q), __uint_as_float(0x3e158106u)), __uint_as_float(0x4564c000u))));
    } else {
        con = pl_fmax(__uint_as_float(0x00000000u), pl_fmul(__uint_as_float(0x3a83126fu), pl_fsub(q, __uint_as_float(0x3f000000u))));
    }
    total = pl_fdiv(pl_fmul(pl_fadd(con, accrete), s[88][0]), s[9][int(s[63][0])]);
    if ((total < s[3][int(s[63][0])])) {
        s[3][int(s[63][0])] = pl_fsub(s[3][int(s[63][0])], total);
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], total);
        return;
    } else {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], s[3][int(s[63][0])]);
        s[3][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:1966-2012.
__device__ void gsl_sublimate(float s[][202]) {
    float dtsubh = 0;
    float dividend = 0;
    float divisor = 0;
    float subl = 0;
    dtsubh = __uint_as_float(0x00000000u);
    if ((s[1][int(s[63][0])] > __uint_as_float(0x4386a666u))) {
        return;
    }
    if ((s[2][int(s[63][0])] <= s[12][int(s[63][0])])) {
        return;
    }
    dividend = pl_fmul(pl_fmul(pl_fmul(pl_fpow(pl_fdiv(__uint_as_float(0x49742400u), s[9][int(s[63][0])]), __uint_as_float(0x3ef33333u)), pl_fsub(pl_fdiv(s[2][int(s[63][0])], s[12][int(s[63][0])]), 1)), pl_fpow(s[5][int(s[63][0])], __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90a3d7u));
    divisor = pl_fadd(__uint_as_float(0x492ae600u), pl_fdiv(__uint_as_float(0x4a7a3e80u), pl_fmul(__uint_as_float(0x41200000u), s[11][int(s[63][0])])));
    dtsubh = pl_fabs(pl_fdiv(dividend, divisor));
    subl = pl_fmul(dtsubh, s[88][0]);
    if ((subl < s[2][int(s[63][0])])) {
        s[2][int(s[63][0])] = pl_fsub(s[2][int(s[63][0])], subl);
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], subl);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(subl, __uint_as_float(0x45306b59u)));
        return;
    } else {
        s[5][int(s[63][0])] = s[2][int(s[63][0])];
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(s[2][int(s[63][0])], __uint_as_float(0x45306b59u)));
        s[2][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:2025-2065.
__device__ void gsl_glaciate(float s[][202]) {
    float dfrzh = 0;
    dfrzh = __uint_as_float(0x00000000u);
    if ((s[4][int(s[63][0])] <= __uint_as_float(0x00000000u))) {
        return;
    }
    if ((s[2][int(s[63][0])] < s[12][int(s[63][0])])) {
        return;
    }
    if ((s[1][int(s[63][0])] > __uint_as_float(0x4386a666u))) {
        return;
    }
    dfrzh = pl_fmul(pl_fmul(s[88][0], __uint_as_float(0x3ccccccdu)), s[4][int(s[63][0])]);
    if ((dfrzh < s[4][int(s[63][0])])) {
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], dfrzh);
        s[4][int(s[63][0])] = pl_fsub(s[4][int(s[63][0])], dfrzh);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a655adu), dfrzh));
        return;
    } else {
        s[5][int(s[63][0])] = pl_fadd(s[5][int(s[63][0])], s[4][int(s[63][0])]);
        s[1][int(s[63][0])] = pl_fadd(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a655adu), s[4][int(s[63][0])]));
        s[4][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:2076-2111.
__device__ void gsl_melt(float s[][202]) {
    float dtmelt = 0;
    dtmelt = __uint_as_float(0x00000000u);
    if ((s[5][int(s[63][0])] <= __uint_as_float(0x00000000u))) {
        return;
    }
    if ((s[1][int(s[63][0])] < __uint_as_float(0x43888000u))) {
        return;
    }
    dtmelt = pl_fmul(pl_fmul(pl_fmul(pl_fmul(pl_fmul(s[88][0], pl_fdiv(__uint_as_float(0x401147aeu), s[9][int(s[63][0])])), s[71][int(s[63][0])]), pl_fsub(s[1][int(s[63][0])], __uint_as_float(0x43888000u))), pl_fpow(pl_fmul(pl_fmul(s[9][int(s[63][0])], s[5][int(s[63][0])]), __uint_as_float(0x358637bdu)), __uint_as_float(0x3f066666u))), __uint_as_float(0x3f90705du));
    if ((dtmelt < s[5][int(s[63][0])])) {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], dtmelt);
        s[5][int(s[63][0])] = pl_fsub(s[5][int(s[63][0])], dtmelt);
        s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a6228fu), dtmelt));
        return;
    } else {
        s[4][int(s[63][0])] = pl_fadd(s[4][int(s[63][0])], s[5][int(s[63][0])]);
        s[1][int(s[63][0])] = pl_fsub(s[1][int(s[63][0])], pl_fmul(__uint_as_float(0x43a6228fu), s[5][int(s[63][0])]));
        s[5][int(s[63][0])] = __uint_as_float(0x00000000u);
    }
    return;
}

// physics/smoke_dust/module_smoke_plumerise.F90:2116-2155.
__device__ void gsl_htint(float s[][202], int nzz1, float* vctra, float* eleva, int nzz2, float* vctrb, float* elevb) {
    char errmsg = 0;
    int errflg = 0;
    int l = 0;
    int k = 0;
    int kk = 0;
    float wt = 0;
    l = 1;
    for (k=1; k<=nzz2; k+=1) {
        while (true) {
            if (((elevb[k] < eleva[1]) || ((elevb[k] >= eleva[l]) && (elevb[k] <= eleva[(l + 1)])))) {
                wt = pl_fdiv(pl_fsub(elevb[k], eleva[l]), pl_fsub(eleva[(l + 1)], eleva[l]));
                vctrb[k] = pl_fadd(vctra[l], pl_fmul(pl_fsub(vctra[(l + 1)], vctra[l]), wt));
                break;
            } else if ((elevb[k] > eleva[nzz1])) {
                wt = pl_fdiv(pl_fsub(elevb[k], eleva[nzz1]), pl_fsub(eleva[(nzz1 - 1)], eleva[nzz1]));
                vctrb[k] = pl_fadd(vctra[nzz1], pl_fmul(pl_fsub(vctra[(nzz1 - 1)], vctra[nzz1]), wt));
                break;
            }
            l = (l + 1);
            if ((l == nzz1)) {
                for (kk=1; kk<=l; kk+=1) {
                }
                ;
                ;
            }
        }
    }
}

// physics/smoke_dust/module_smoke_plumerise.F90:2162-2185.
__device__ float gsl_esat_pr(float s[][202], float tem) {
    float esat_pr = 0;
    float temc = 0;
    float esatm = 0;
    temc = pl_fsub(tem, __uint_as_float(0x4388a666u));
    if ((temc <= __uint_as_float(0xc2200000u))) {
        esatm = pl_fmul(__uint_as_float(0x40c39168u), pl_fexp(pl_fdiv(pl_fmul(__uint_as_float(0x41b45604u), temc), pl_fadd(temc, __uint_as_float(0x4388bd71u)))));
        esat_pr = pl_fdiv(esatm, __uint_as_float(0x41200000u));
        return esat_pr;
    }
    esatm = pl_fmul(__uint_as_float(0x40c39653u), pl_fexp(pl_fdiv(pl_fmul(pl_fsub(__uint_as_float(0x4195d4feu), pl_fdiv(temc, __uint_as_float(0x43634ccdu))), temc), pl_fadd(temc, __uint_as_float(0x4380ef5cu)))));
    esat_pr = pl_fdiv(esatm, __uint_as_float(0x41200000u));
    return esat_pr;
}

extern "C" __global__ void ebu_distribute(
    const int *columns, const int *kp1, const int *kp2,
    const float *flam_frac, const float *ebu_in, const float *z_at_w,
    float *ebu, int nfire, int nz, int nxy, int nrows)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= nfire) return;
    int ij = columns[c];
    int lo = kp1[ij] - 1, hi = kp2[ij] - 1;
    float frac = flam_frac[ij];
    float dz = __fsub_rn(z_at_w[hi*nxy+ij], z_at_w[lo*nxy+ij]);
    for (int r=0; r<nrows; ++r) {
        float incoming = ebu_in[r*nxy+ij];
        for (int k=0; k<nz; ++k) {
            float value = 0.0f;
            if (k >= lo && k < hi) {
                float thickness = __fsub_rn(z_at_w[(k+1)*nxy+ij], z_at_w[k*nxy+ij]);
                value = __fdiv_rn(__fmul_rn(__fmul_rn(frac,incoming),thickness),dz);
            }
            if (k == 0) value = __fmul_rn(__fsub_rn(1.0f,frac),incoming);
            ebu[(r*nz+k)*nxy+ij] = value;
        }
    }
}

// Inputs: [fire_column, eight met fields, nz+1], row vectors and column properties.
// Output profiles use one-based GSL bounds externally and the model's mass levels.
extern "C" __global__ void freitas_columns(
    const float* met, const float* input_emissions, const float* properties,
    const float* params, float* emissions, int* bounds, float* diagnostics,
    float* profiles, float* group_data, int* iterations, int* status, int nfire, int nz, int nrows, int arm,
    float* workspace, int first, int count)
{
    // One launch covers fire columns first..first+count-1 (the host batches
    // the fire columns so the workspace stays bounded); the workspace holds
    // `count` column slices.
    int c=first+blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nfire || c>=first+count) return;
    status[c]=0;
    // The plume state lives in the column's own slice of a global workspace
    // (PL_WS_ROWS x 202 words), not in local memory: 114 x 202 floats of
    // local state were a 93 KB frame, and CUDA reserves a kernel's frame for
    // every resident thread of the card at once (24 GB on a 5090).  Rows
    // 0..113 are the transcription's state, 114..118 the heat table, 119 the
    // plume tops, 120.. the routines' own 202-word scratch.  Zeroed here as
    // `= {}` zeroed the locals, so every word read is one this column wrote.
    float (*s)[202] = reinterpret_cast<float (*)[202]>(workspace + (size_t)(c - first) * PL_WS_ROWS * 202);
    for (int _r=0; _r<PL_WS_ROWS; ++_r) for (int _z=0; _z<202; ++_z) s[_r][_z] = 0.0f;
    float (*heat)[202] = s + 114;
    for(int group=1;group<=4;++group) { heat[group][1]=params[7+(group-1)*2]; heat[group][2]=params[8+(group-1)*2]; }
    int stride=nz+1;
    const float* m=met+c*8*stride;
    const float* prop=properties+c*12;
    const float* incoming=input_emissions+c*nrows;
    float* out=emissions+c*nrows*nz;
    float cp=__uint_as_float(0x447b2000u), rd=287.0f;
    float rcp=FDIV(rd,cp);
    for(int k=1;k<=nz;++k) {
        float pi=FMUL(cp,gfk_pow(FDIV(m[stride+k-1],100000.0f),rcp));
        s[49][k]=pi;
        s[47][k]=FMUL(FDIV(m[k-1],pi),cp);
        s[48][k]=m[3*stride+k-1];
        s[44][k]=m[4*stride+k-1]; s[45][k]=m[5*stride+k-1];
        s[53][k]=FSUB(m[6*stride+k-1],m[7*stride]);
        s[54][k]=FSUB(m[7*stride+k-1],m[7*stride]);
    }
    int kmt=0;
    if(arm==0) gsl_get_env_condition(s,1,nz,kmt,int(params[6]),9.81f,cp,rd,FDIV(cp,rd));
    else wrf_get_env_condition(s,1,nz,kmt,1);
    for(int r=0;r<nrows;++r) for(int k=0;k<nz;++k) out[r*nz+k]=0.0f;
    int lo=1,hi=2;
    float fraction=0.0f,topmax=0.0f;
    for(int group=0;group<4;++group) {
        diagnostics[c*10+1+group*2]=0.0f; diagnostics[c*10+2+group*2]=0.0f;
        iterations[c*8+group*2]=0; iterations[c*8+group*2+1]=0;
        for(int f=0;f<4;++f) group_data[c*16+group*4+f]=0.0f;
    }
    if(arm==0) {
        float frp=prop[0];
        if(frp>=params[0]) {
            float* tt = s[119];  // workspace row 119: the two plume tops of this column
            for (int _z=0; _z<202; ++_z) tt[_z] = 0.0f;
            for(int imm=1;imm<=2;++imm) {
                float coefficient=__uint_as_float(imm==1?0x39dc3373u:0x3a4c78eau);
                float area=fmaxf(10000.0f,FMUL(coefficient,frp));
                gsl_get_fire_properties(s,imm,area,frp);
                float top=0.0f;
                gsl_makeplume(s,kmt,top,0,imm,0,params[5]);
                tt[imm]=top; topmax=fmaxf(topmax,top);
                diagnostics[c*10+imm]=top;
                iterations[c*8+imm-1]=int(s[108][0]);
            }
            gsl_set_flam_vert(s,tt,lo,hi,200,s[54]);
        }
        if(frp<=params[0]) fraction=0.0f;
        else if(frp<=params[1] && prop[10]>=params[3] && prop[11]>params[2] && int(params[6])==1) {
            lo=2; hi=max(3,int(floorf(FADD(FDIV(prop[9],3.0f),0.5f))));
            fraction=__uint_as_float(0x3f59999au);
        } else fraction=__uint_as_float(0x3f666666u);
        if(lo<1 || hi<=lo || hi>nz+1) { status[c]=1; return; }
        float depth=FSUB(m[7*stride+hi-1],m[7*stride+lo-1]);
        for(int r=0;r<nrows;++r) {
            for(int k=lo;k<hi;++k) out[r*nz+k-1]=FDIV(FMUL(FMUL(fraction,incoming[r]),FSUB(m[7*stride+k],m[7*stride+k-1])),depth);
            out[r*nz]=FMUL(FSUB(1.0f,fraction),incoming[r]);
        }
    } else {
        for(int r=0;r<nrows;++r) out[r*nz]=incoming[r];
        float sf=FADD(FADD(FADD(prop[1],prop[2]),prop[3]),prop[4]);
        float ss=FADD(FADD(FADD(prop[5],prop[6]),prop[7]),prop[8]);
        float maximum=0.0f; for(int r=0;r<nrows;++r) maximum=fmaxf(maximum,incoming[r]);
        if(sf>=1.e-6f && ss>=1.e-6f && maximum!=0.0f) {
            lo=200; hi=1;
            for(int group=1;group<=4;++group) {
                if(prop[group]<1.e-6f) continue;
                float* tt = s[119];  // workspace row 119: the two plume tops of this column
                for (int _z=0; _z<202; ++_z) tt[_z] = 0.0f;
                for(int imm=1;imm<=2;++imm) {
                    wrf_get_fire_properties(s,imm,group,prop[4+group],0.0f,heat);
                    if(params[15+group]!=0.0f && imm==2) { tt[2]=tt[1]; tt[1]=s[54][1]; }
                    else {
                        wrf_makeplume(s,kmt,tt[imm],0,imm);
                        iterations[c*8+(group-1)*2+imm-1]=int(s[108][0]);
                    }
                }
                diagnostics[c*10+1+(group-1)*2]=tt[1]; diagnostics[c*10+2+(group-1)*2]=tt[2];
                topmax=fmaxf(topmax,fmaxf(tt[1],tt[2]));
                int k1=0,k2=0;
                wrf_set_flam_vert(s,tt,k1,k2,200,s[54],s+94,s+97);
                if(k1<1 || k2<k1 || k2>=nz) { status[c]=1; return; }
                lo=min(lo,k1); hi=max(hi,k2+1);
                float inv=FDIV(1.0f,FSUB(s[54][k2+1],s[54][k1]));
                group_data[c*16+(group-1)*4]=float(k1);
                group_data[c*16+(group-1)*4+1]=float(k2);
                group_data[c*16+(group-1)*4+2]=prop[group];
                group_data[c*16+(group-1)*4+3]=inv;
                for(int k=k1;k<=k2;++k) for(int r=0;r<nrows;++r)
                    out[r*nz+k-1]=FADD(out[r*nz+k-1],FMUL(FMUL(prop[group],incoming[r]),inv));
            }
            for(int k=1;k<nz;++k) for(int r=0;r<nrows;++r)
                out[r*nz+k]=FMUL(out[r*nz+k],FSUB(m[7*stride+k+1],m[7*stride+k]));
            fraction=sf;
        } else {
            for(int row=0;row<114;++row) for(int k=0;k<202;++k) s[row][k]=0.0f;
        }
    }
    bounds[c*2]=lo; bounds[c*2+1]=hi;
    diagnostics[c*10]=fraction;
    // PLUME_TOP is the larger top above ground, across all active groups.
    // Plume profiles are optional audit outputs, never forecast allocations.
    const int slots[7]={0,1,2,3,4,5,84};
    if(profiles) for(int f=0;f<7;++f) for(int k=1;k<=200;++k)
        profiles[(c*7+f)*200+k-1]=s[slots[f]][k];
    diagnostics[c*10+9]=topmax;
}

// WRF inclusive group bounds and left-to-right accumulation preserve species independence.
extern "C" __global__ void ebu_distribute_landuse(
    const int* columns,const float* cache,const float* incoming,
    const float* z_at_w,float* ebu,int nfire,int nz,int nxy,int nrows)
{
    int c=blockDim.x*blockIdx.x+threadIdx.x; if(c>=nfire) return;
    int ij=columns[c];
    for(int r=0;r<nrows;++r) {
        float value=incoming[r*nxy+ij];
        ebu[(r*nz)*nxy+ij]=value;
        for(int k=1;k<nz;++k) {
            float density=0.0f;
            for(int g=0;g<4;++g) {
                int lo=int(cache[(g*4)*nxy+ij]);
                int hi=int(cache[(g*4+1)*nxy+ij]);
                float fraction=cache[(g*4+2)*nxy+ij];
                if(fraction<1.e-6f || k+1<lo || k+1>hi) continue;
                float inv=FDIV(1.0f,FSUB(z_at_w[hi*nxy+ij],z_at_w[(lo-1)*nxy+ij]));
                density=FADD(density,FMUL(FMUL(fraction,value),inv));
            }
            ebu[(r*nz+k)*nxy+ij]=FMUL(density,FSUB(z_at_w[(k+1)*nxy+ij],z_at_w[k*nxy+ij]));
        }
    }
}
