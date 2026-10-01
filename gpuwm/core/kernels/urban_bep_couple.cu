// BEP / BEP+BEM surface coupling (sf_urban_physics 2 and 3), WRF v4.7.1.
//
// urban_bep_couple  -- what the LSM driver does to BEP's output before the
//   PBL sees it, one thread per column over EVERY column (WRF runs this
//   block over the whole tile, urban or not: a rural column still gets its
//   own surface flux folded into level 1, which is how YSU/MYJURB receive
//   the surface flux at all once flag_bep is on):
//     Noah     module_sf_noahdrv.F:1679-1776
//     Noah-MP  module_sf_noahmpdrv.F:3689-3776 (rural words copied from the
//              LSM's fields at :3363-3372)
//   The two blocks differ in exactly two statements, selected by `lsm`
//   (2 = Noah, 4 = Noah-MP):
//     grdflx   Noah-MP flips the sign of the urban ground flux (:3752)
//     lh_urb2d Noah-MP divides by frc (:3756)
// urban_bep_sfcdiag -- module_surface_driver.F:3022-3035 (Noah) ==
//   :3408-3421 (Noah-MP): on urban-category columns T2/TH2/Q2/U10/V10 are
//   the lowest model level, overriding SFCDIAGS / Noah-MP's 2 m values.
//
// Arithmetic is WRF's -r4 as gfortran -O0 evaluates it: left to right,
// every product rounded before it is added (FMUL/FADD from glibc_flt32.cuh;
// NVRTC would otherwise fuse them), and every REAL-constant power (`**.5`,
// `**2.`, `**4.`, `**.25`, `**RCP`) is glibc's powf -- measured: gfortran
// -O0 emits 14 powf calls for each block and turns only the INTEGER `**2`
// into a multiply.
//
// Layout: (k, ny, nx), x fastest; 2-D fields (ny, nx).  sf_bep has nz+1
// levels (WRF's kms:kme); the couple touches its first nz, as WRF does.

#define UBC_SIGMA_SB 5.67e-08f      // noahdrv.F:727, noahmpdrv.F:3691
// module_model_constants.F: r_d = 287., cp = 7.*r_d/2. (= 1004.5 exactly),
// xlv = 2.5E6, rcp = r_d/cp.
#define UBC_CP 1004.5f
#define UBC_XLV 2.5e6f
#define UBC_RCP (287.0f / 1004.5f)

extern "C" __global__
void urban_bep_couple(const float *frc_urb2d, const float *dz8w,
                      const float *rho, const float *u_phy,
                      const float *v_phy, const float *glw,
                      const float *swdown,
                      // BEP's column outputs (zeroed then written by BEP)
                      const float *rl_up_urb, const float *rs_abs_urb,
                      const float *emiss_urb, const float *grdflx_urb,
                      // PBL handoff, weighted in place
                      float *a_u_bep, float *a_v_bep, float *a_t_bep,
                      float *a_q_bep, float *a_e_bep, float *b_u_bep,
                      float *b_v_bep, float *b_t_bep, float *b_q_bep,
                      float *b_e_bep, float *sf_bep, float *vl_bep,
                      // LSM fields: read as the rural words, then blended
                      float *ust, float *tsk, float *hfx, float *qfx,
                      float *lh, float *grdflx, float *albedo, float *emiss,
                      // urban diagnostics written on frc > 0 columns
                      float *ts_urb2d, float *sh_urb2d, float *lh_urb2d,
                      float *g_urb2d, float *rn_urb2d,
                      int lsm, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st) return;

    // Rural words.  Noah: HFX_RURAL=SHEAT etc. are the same words the LSM
    // wrote to HFX etc. (noahdrv.F:1225-1273, and :871-876 for water);
    // Noah-MP copies them from the fields (noahmpdrv.F:3363-3372).
    const float hfx_rural = hfx[col];
    const float qfx_rural = qfx[col];
    const float grdflx_rural = grdflx[col];
    const float emiss_rural = emiss[col];
    const float tsk_rural = tsk[col];
    const float alb_rural = albedo[col];
    const float frc = frc_urb2d[col];

    float umom_urb = 0.0f, vmom_urb = 0.0f, hfx_urb = 0.0f, qfx_urb = 0.0f;
    for (int k = 0; k < nz; ++k) {                              // :1685
        int q = k * st + col;
        a_u_bep[q] = FMUL(a_u_bep[q], frc);
        a_v_bep[q] = FMUL(a_v_bep[q], frc);
        a_t_bep[q] = FMUL(a_t_bep[q], frc);
        a_q_bep[q] = 0.0f;
        a_e_bep[q] = 0.0f;
        b_u_bep[q] = FMUL(b_u_bep[q], frc);
        b_v_bep[q] = FMUL(b_v_bep[q], frc);
        b_t_bep[q] = FMUL(b_t_bep[q], frc);
        b_q_bep[q] = FMUL(b_q_bep[q], frc);
        b_e_bep[q] = FMUL(b_e_bep[q], frc);
        // :1696 HFX_URB + B_T*RHO*CP*DZ8W*VL
        hfx_urb = FADD(hfx_urb, FMUL(FMUL(FMUL(FMUL(b_t_bep[q], rho[q]),
                                               UBC_CP), dz8w[q]), vl_bep[q]));
        // :1698 QFX_URB + B_Q*DZ8W*VL
        qfx_urb = FADD(qfx_urb, FMUL(FMUL(b_q_bep[q], dz8w[q]), vl_bep[q]));
        // :1700 UMOM_URB + (A_U*U+B_U)*DZ8W*VL
        umom_urb = FADD(umom_urb,
                        FMUL(FMUL(FADD(FMUL(a_u_bep[q], u_phy[q]), b_u_bep[q]),
                                  dz8w[q]), vl_bep[q]));
        vmom_urb = FADD(vmom_urb,
                        FMUL(FMUL(FADD(FMUL(a_v_bep[q], v_phy[q]), b_v_bep[q]),
                                  dz8w[q]), vl_bep[q]));
        // :1704-1705
        vl_bep[q] = FADD(FSUB(1.0f, frc), FMUL(vl_bep[q], frc));
        sf_bep[q] = FADD(FSUB(1.0f, frc), FMUL(sf_bep[q], frc));
    }

    const float u1 = u_phy[col], v1 = v_phy[col];
    const float dz1 = dz8w[col], rho1 = rho[col];
    const float us = ust[col];
    const float omf = FSUB(1.0f, frc);
    // ((u**2 + v**2.)**.5): integer square, REAL-constant square, powf .5
    const float spd = gfk_pow(FADD(FMUL(u1, u1), gfk_pow(v1, 2.0f)), 0.5f);
    // :1707 (1.-frc)*(-ust*ust)/dz8w/(spd)+a_u
    a_u_bep[col] = FADD(FDIV(FDIV(FMUL(omf, FMUL(-us, us)), dz1), spd),
                        a_u_bep[col]);
    const float spd2 = gfk_pow(FADD(FMUL(u1, u1), gfk_pow(v1, 2.0f)), 0.5f);
    a_v_bep[col] = FADD(FDIV(FDIV(FMUL(omf, FMUL(-us, us)), dz1), spd2),
                        a_v_bep[col]);
    // :1711 (1.-frc)*hfx_rural/dz8w/rho/CP + b_t
    b_t_bep[col] = FADD(FDIV(FDIV(FDIV(FMUL(omf, hfx_rural), dz1), rho1),
                             UBC_CP), b_t_bep[col]);
    // :1714 (1.-frc)*qfx_rural/dz8w/rho + b_q
    b_q_bep[col] = FADD(FDIV(FDIV(FMUL(omf, qfx_rural), dz1), rho1),
                        b_q_bep[col]);
    // :1715 (1.-frc)*ust*ust*u/(spd)+umom_urb
    const float spd3 = gfk_pow(FADD(FMUL(u1, u1), gfk_pow(v1, 2.0f)), 0.5f);
    const float umom = FADD(FDIV(FMUL(FMUL(FMUL(omf, us), us), u1), spd3),
                            umom_urb);
    const float spd4 = gfk_pow(FADD(FMUL(u1, u1), gfk_pow(v1, 2.0f)), 0.5f);
    const float vmom = FADD(FDIV(FMUL(FMUL(FMUL(omf, us), us), v1), spd4),
                            vmom_urb);
    sf_bep[col] = 1.0f;                                         // :1719

    if (frc > 0.0f) {                                           // :1726
        const float g = glw[col], sw = swdown[col];
        const float eu = emiss_urb[col], rlu = rl_up_urb[col];
        // -emiss_rural*sigma_sb*(tsk_rural**4.)-(1.-emiss_rural)*glw
        const float rl_up_rural =
            FSUB(FMUL(FMUL(-emiss_rural, UBC_SIGMA_SB),
                      gfk_pow(tsk_rural, 4.0f)),
                 FMUL(FSUB(1.0f, emiss_rural), g));
        const float rl_up_tot = FADD(FMUL(omf, rl_up_rural), FMUL(frc, rlu));
        const float em = FADD(FMUL(omf, emiss_rural), FMUL(frc, eu));
        emiss[col] = em;
        // (max(0.,(-rl_up_urb-(1.-emiss_urb)*glw)/emiss_urb/sigma_sb))**0.25
        float x = FDIV(FDIV(FSUB(-rlu, FMUL(FSUB(1.0f, eu), g)), eu),
                       UBC_SIGMA_SB);
        ts_urb2d[col] = gfk_pow(fmaxf(0.0f, x), 0.25f);
        // (max(0., (-1.*rl_up_tot-(1.-emiss)*glw )/emiss/sigma_sb))**.25
        x = FDIV(FDIV(FSUB(FMUL(-1.0f, rl_up_tot), FMUL(FSUB(1.0f, em), g)),
                      em), UBC_SIGMA_SB);
        tsk[col] = gfk_pow(fmaxf(0.0f, x), 0.25f);
        // (1.-frc)*swdown*(1.-albedo)+frc*rs_abs_urb
        const float rs_abs_tot =
            FADD(FMUL(FMUL(omf, sw), FSUB(1.0f, albedo[col])),
                 FMUL(frc, rs_abs_urb[col]));
        if (sw > 0.0f) albedo[col] = FSUB(1.0f, FDIV(rs_abs_tot, sw));
        else albedo[col] = alb_rural;
        const float gu = grdflx_urb[col];
        if (lsm == 4)                                   // noahmpdrv.F:3752
            grdflx[col] = FADD(FMUL(omf, grdflx_rural),
                               FMUL(FMUL(frc, gu), -1.0f));
        else                                            // noahdrv.F:1739
            grdflx[col] = FADD(FMUL(omf, grdflx_rural), FMUL(frc, gu));
        const float qf = FADD(FMUL(omf, qfx_rural), qfx_urb);
        qfx[col] = qf;
        lh[col] = FMUL(qf, UBC_XLV);
        // HFX_URB+(1-FRC)*HFX_RURAL
        hfx[col] = FADD(hfx_urb, FMUL(omf, hfx_rural));
        sh_urb2d[col] = FDIV(hfx_urb, frc);
        if (lsm == 4)                                   // noahmpdrv.F:3756
            lh_urb2d[col] = FDIV(FMUL(qfx_urb, UBC_XLV), frc);
        else                                            // noahdrv.F:1745
            lh_urb2d[col] = FMUL(qfx_urb, UBC_XLV);
        g_urb2d[col] = gu;
        // rs_abs_urb+emiss_urb*glw-rl_up_urb
        rn_urb2d[col] = FSUB(FADD(rs_abs_urb[col], FMUL(eu, g)), rlu);
        // (umom**2.+vmom**2.)**.25
        ust[col] = gfk_pow(FADD(gfk_pow(umom, 2.0f), gfk_pow(vmom, 2.0f)),
                           0.25f);
    } else {
        sh_urb2d[col] = 0.0f;
        lh_urb2d[col] = 0.0f;
        g_urb2d[col] = 0.0f;
        rn_urb2d[col] = 0.0f;
    }
}

// module_surface_driver.F:3022-3035 / :3408-3421.  `utype_urb2d > 0` is
// exactly urban_var_init's IVGTYP in {ISURBAN, LCZ_1..LCZ_11}
// (module_sf_urban.F:2757-2786), the set the surface driver tests.
extern "C" __global__
void urban_bep_sfcdiag(const int *utype_urb2d, const float *th_phy,
                       const float *qv, const float *u_phy,
                       const float *v_phy, const float *psfc,
                       float *t2, float *th2, float *q2, float *u10,
                       float *v10, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx || utype_urb2d[col] <= 0) return;
    const float th1 = th_phy[col];
    // TH_PHY(i,1,j)/((1.E5/PSFC(I,J))**RCP)
    t2[col] = FDIV(th1, gfk_pow(FDIV(1.0e5f, psfc[col]), UBC_RCP));
    th2[col] = th1;
    q2[col] = qv[col];
    u10[col] = u_phy[col];
    v10[col] = v_phy[col];
}
