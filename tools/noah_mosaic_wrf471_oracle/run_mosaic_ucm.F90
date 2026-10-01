program run_mosaic_ucm
  ! WRF v4.7.1 public entry points only. No expected value is transcribed.
  ! One process per case avoids saved BEP parameters and allocated UCM tables
  ! leaking between options. run_all invokes this executable in child mode.
  use module_sf_noahdrv, only: lsm_mosaic, lsm_mosaic_init, soil_veg_gen_parm
  use module_sf_noahlsm, only: LCZ_1,LCZ_2,LCZ_3,LCZ_4,LCZ_5,LCZ_6, &
       LCZ_7,LCZ_8,LCZ_9,LCZ_10,LCZ_11,NATURAL
  use module_sf_urban, only: urban_param_init, urban_var_init, &
       ch_scheme_data, ts_scheme_data, ahoption, alhoption, imp_scheme, iri_scheme, groption, &
       ZR_TBL, Z0C_TBL, Z0HC_TBL, ZDC_TBL, SVF_TBL, &
       R_TBL, RW_TBL, HGT_TBL, AH_TBL, ALH_TBL, &
       BETR_TBL, BETB_TBL, BETG_TBL, CAPR_TBL, CAPB_TBL, &
       CAPG_TBL, AKSR_TBL, AKSB_TBL, AKSG_TBL, ALBR_TBL, &
       ALBB_TBL, ALBG_TBL, EPSR_TBL, EPSB_TBL, EPSG_TBL, &
       Z0R_TBL, Z0B_TBL, Z0G_TBL, Z0HB_TBL, Z0HG_TBL, &
       TRLEND_TBL, TBLEND_TBL, TGLEND_TBL, AKANDA_URBAN_TBL, frc_urb_tbl, &
       boundr_data, boundb_data, boundg_data, oasis, dzgr, &
       porimp, dengimp, ahdiuprf, alhseason, alhdiuprf, &
       fgr
  use module_bep_bem_helper, only: nurbm
  use oracle_io
  implicit none
  integer, parameter :: ncase=48, nsoil=4, nlev=2, isurban=13
  real, parameter :: dt=60.0, r_d_over_cp=287.0/(7.0*287.0/2.0)
  integer, parameter :: num_urban_ndm=2
  integer, parameter :: urban_map_zrd=360
  integer, parameter :: urban_map_zwd=360
  integer, parameter :: urban_map_gd=20
  integer, parameter :: urban_map_zd=36
  integer, parameter :: urban_map_zdf=36
  integer, parameter :: urban_map_bd=36
  integer, parameter :: urban_map_wd=36
  integer, parameter :: urban_map_gbd=2
  integer, parameter :: urban_map_fbd=20
  integer, parameter :: urban_map_zgrd=360
  integer, parameter :: num_urban_hi=15
  ! --- driver argument arrays, one column per case ---------------------------
  real, dimension(ncase, nlev, 1) :: dz8w, qv3d, p8w3d, t3d
  real, dimension(ncase, 1) :: tsk, hfx, qfx, lh, grdflx, qgh, gsw, swdown
  real, dimension(ncase, 1) :: swddir, swddif, glw, smstav, smstot
  real, dimension(ncase, 1) :: sfcrunoff, udrunoff, vegfra, albedo, albbck
  real, dimension(ncase, 1) :: znt, z0, tmn, xland, xice, emiss, embck
  real, dimension(ncase, 1) :: snowc, qsfc, rainbl, snow, canwat
  real, dimension(ncase, 1) :: chs, chs2, cqs2, cpm, sr, chklowq, lai, qz0
  real, dimension(ncase, 1) :: snowh, snoalb, shdmin, shdmax, snotime
  real, dimension(ncase, 1) :: acsnom, acsnow, snopcx, potevp, rib, noahres
  real, dimension(ncase, 1) :: flx4_2d, fvb_2d, fbur_2d, fgsn_2d, ust_urb2d
  real, dimension(ncase, 1) :: frc_urb2d
  real, dimension(ncase, 1) :: sda_hfx, sda_qfx, hfx_both, qfx_both, qnorm
  real, dimension(ncase, nsoil, 1) :: smois, tslb, sh2o, smcrel
  integer, dimension(ncase, 1) :: ivgtyp, isltyp, utype_urb2d
  real, dimension(nsoil) :: dzs

  real, dimension( ncase, 1 ) :: tr_urb2d
  real, dimension( ncase, 1 ) :: tb_urb2d
  real, dimension( ncase, 1 ) :: tg_urb2d
  real, dimension( ncase, 1 ) :: tc_urb2d
  real, dimension( ncase, 1 ) :: qc_urb2d
  real, dimension( ncase, 1 ) :: xxxr_urb2d
  real, dimension( ncase, 1 ) :: xxxb_urb2d
  real, dimension( ncase, 1 ) :: xxxg_urb2d
  real, dimension( ncase, 1 ) :: xxxc_urb2d
  real, dimension( ncase, 1 ) :: drelr_urb2d
  real, dimension( ncase, 1 ) :: drelb_urb2d
  real, dimension( ncase, 1 ) :: drelg_urb2d
  real, dimension( ncase, 1 ) :: flxhumr_urb2d
  real, dimension( ncase, 1 ) :: flxhumb_urb2d
  real, dimension( ncase, 1 ) :: flxhumg_urb2d
  real, dimension( ncase, 1 ) :: cmcr_urb2d
  real, dimension( ncase, 1 ) :: tgr_urb2d
  real, dimension(ncase, nsoil, 1) :: trl_urb3d
  real, dimension(ncase, nsoil, 1) :: tbl_urb3d
  real, dimension(ncase, nsoil, 1) :: tgl_urb3d
  real, dimension(ncase, nsoil, 1) :: tgrl_urb3d
  real, dimension(ncase, nsoil, 1) :: smr_urb3d
  real, dimension( ncase, 1 ) :: sh_urb2d
  real, dimension( ncase, 1 ) :: lh_urb2d
  real, dimension( ncase, 1 ) :: g_urb2d
  real, dimension( ncase, 1 ) :: rn_urb2d
  real, dimension( ncase, 1 ) :: ts_urb2d
  real, dimension(ncase, 1:urban_map_zrd, 1) :: trb_urb4d
  real, dimension(ncase, 1:urban_map_zwd, 1) :: tw1_urb4d
  real, dimension(ncase, 1:urban_map_zwd, 1) :: tw2_urb4d
  real, dimension(ncase, 1:urban_map_gd , 1) :: tgb_urb4d
  real, dimension(ncase, 1:urban_map_bd , 1) :: tlev_urb3d
  real, dimension(ncase, 1:urban_map_bd , 1) :: qlev_urb3d
  real, dimension(ncase, 1:urban_map_wd , 1) :: tw1lev_urb3d
  real, dimension(ncase, 1:urban_map_wd , 1) :: tw2lev_urb3d
  real, dimension(ncase, 1:urban_map_gbd, 1) :: tglev_urb3d
  real, dimension(ncase, 1:urban_map_fbd, 1) :: tflev_urb3d
  real, dimension( ncase, 1 ) :: lf_ac_urb3d
  real, dimension( ncase, 1 ) :: sf_ac_urb3d
  real, dimension( ncase, 1 ) :: cm_ac_urb3d
  real, dimension( ncase, 1 ) :: sfvent_urb3d
  real, dimension( ncase, 1 ) :: lfvent_urb3d
  real, dimension( ncase, 1:urban_map_wd, 1) :: sfwin1_urb3d
  real, dimension( ncase, 1:urban_map_wd, 1) :: sfwin2_urb3d
  real, dimension(ncase, 1:urban_map_zd , 1) :: sfw1_urb3d
  real, dimension(ncase, 1:urban_map_zd , 1) :: sfw2_urb3d
  real, dimension(ncase, 1:urban_map_zdf, 1) :: sfr_urb3d
  real, dimension(ncase, 1:num_urban_ndm, 1) :: sfg_urb3d
  real, dimension( ncase, 1 ) :: ep_pv_urb3d
  real, dimension( ncase, 1:urban_map_zdf,1 ) :: t_pv_urb3d
  real, dimension( ncase, 1:urban_map_zgrd, 1) :: trv_urb4d
  real, dimension( ncase, 1:urban_map_zgrd, 1) :: qr_urb4d
  real, dimension( ncase,1) :: qgr_urb3d
  real, dimension( ncase,1) :: tgr_urb3d
  real, dimension( ncase, 1:urban_map_zdf, 1) :: drain_urb4d
  real, dimension( ncase, 1 ) :: draingr_urb3d
  real, dimension( ncase, 1:urban_map_zdf, 1) :: sfrv_urb3d
  real, dimension( ncase, 1:urban_map_zdf, 1) :: lfrv_urb3d
  real, dimension( ncase, 1:urban_map_zdf, 1 ) :: dgr_urb3d
  real, dimension( ncase, 1:num_urban_ndm, 1 ) :: dg_urb3d
  real, dimension( ncase, 1:urban_map_zdf, 1 ) :: lfr_urb3d
  real, dimension( ncase, 1:num_urban_ndm, 1 ) :: lfg_urb3d
  real, dimension( ncase,1:num_urban_hi , 1) :: hi_urb2d
  real, dimension( ncase, 1 ) :: lp_urb2d
  real, dimension( ncase, 1 ) :: lb_urb2d
  real, dimension( ncase, 1 ) :: hgt_urb2d
  real, dimension( ncase, 1 ) :: mh_urb2d
  real, dimension( ncase, 1 ) :: stdh_urb2d
  real, dimension( ncase, 4,1 ) :: lf_urb2d
  real, dimension(ncase, nlev, 1) :: a_u_bep
  real, dimension(ncase, nlev, 1) :: a_v_bep
  real, dimension(ncase, nlev, 1) :: a_t_bep
  real, dimension(ncase, nlev, 1) :: a_q_bep
  real, dimension(ncase, nlev, 1) :: a_e_bep
  real, dimension(ncase, nlev, 1) :: b_u_bep
  real, dimension(ncase, nlev, 1) :: b_v_bep
  real, dimension(ncase, nlev, 1) :: b_t_bep
  real, dimension(ncase, nlev, 1) :: b_q_bep
  real, dimension(ncase, nlev, 1) :: b_e_bep
  real, dimension(ncase, nlev, 1) :: vl_bep
  real, dimension(ncase, nlev, 1) :: dlg_bep
  real, dimension(ncase, nlev,1) :: sf_bep
  real, dimension(ncase, nlev, 1) :: dl_u_bep
  real, dimension(ncase, 1) :: uc_urb2d
  real, dimension(ncase, 1) :: psim_urb2d
  real, dimension(ncase, 1) :: psih_urb2d
  real, dimension(ncase, 1) :: u10_urb2d
  real, dimension(ncase, 1) :: v10_urb2d
  real, dimension(ncase, 1) :: gz1oz0_urb2d
  real, dimension(ncase, 1) :: akms_urb2d
  real, dimension(ncase, 1) :: th2_urb2d
  real, dimension(ncase, 1) :: q2_urb2d
  real, dimension(ncase, 1) :: cosz_urb2d
  real, dimension(ncase, 1) :: omg_urb2d
  real, dimension(ncase, 1) :: xlat_urb2d
  real, dimension(ncase, 1) :: lf_urb2d_s
  real, dimension(ncase, 1) :: z0_urb2d
  real, dimension(ncase, 1) :: ust
  real, dimension(ncase, 1) :: tsk_rural_bep
  real, dimension(ncase, 1) :: xlong
  real, dimension(ncase, 1) :: xlat
  real, dimension(ncase, nlev, 1) :: u_phy
  real, dimension(ncase, nlev, 1) :: v_phy
  real, dimension(ncase, nlev, 1) :: th_phy
  real, dimension(ncase, nlev, 1) :: rho
  real, dimension(ncase, nlev, 1) :: p_phy
  real :: cmr_sfcdif(ncase,1)
  real :: chr_sfcdif(ncase,1)
  real :: cmc_sfcdif(ncase,1)
  real :: chc_sfcdif(ncase,1)
  real :: cmgr_sfcdif(ncase,1)
  real :: chgr_sfcdif(ncase,1)
  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil)
  integer :: i, j, k, ns, sf_urban_physics, lcz, opt_thcnd, itimestep, status
  logical :: usemonalb, frpcpn, rdlai2d
  character(len=1024) :: outdir, exe
  character(len=40) :: case_name
  real :: rc2(ncase,1),xlai2(ncase,1)
  integer :: mc, nlcat, switch, step, t, twin, use_lcz
  ! Families 3 and 4 repeat 1 and 2 with FRC_URB2D exactly as urban_var_init
  ! leaves it: the table fraction where the DOMINANT category is urban and
  ! zero elsewhere (module_sf_urban.F:2811-2821), WRF's own urban rule.
  logical :: wrf_init_frc
  character(len=16) :: arg
  real :: declin=0.2
  real :: u10(ncase,1),v10(ncase,1),psim(ncase,1),psih(ncase,1),gz1oz0(ncase,1),akhs(ncase,1),akms(ncase,1)
  real :: subn,nzero
  real, allocatable :: landusef(:,:,:),landusef2(:,:,:)
  integer, allocatable :: mosaic_cat_index(:,:,:)
  real, allocatable :: tsk_mosaic(:,:,:)
  real, allocatable :: qsfc_mosaic(:,:,:)
  real, allocatable :: canwat_mosaic(:,:,:)
  real, allocatable :: snow_mosaic(:,:,:)
  real, allocatable :: snowh_mosaic(:,:,:)
  real, allocatable :: snowc_mosaic(:,:,:)
  real, allocatable :: albedo_mosaic(:,:,:)
  real, allocatable :: albbck_mosaic(:,:,:)
  real, allocatable :: emiss_mosaic(:,:,:)
  real, allocatable :: embck_mosaic(:,:,:)
  real, allocatable :: znt_mosaic(:,:,:)
  real, allocatable :: z0_mosaic(:,:,:)
  real, allocatable :: hfx_mosaic(:,:,:)
  real, allocatable :: qfx_mosaic(:,:,:)
  real, allocatable :: lh_mosaic(:,:,:)
  real, allocatable :: grdflx_mosaic(:,:,:)
  real, allocatable :: snotime_mosaic(:,:,:)
  real, allocatable :: rc_mosaic(:,:,:)
  real, allocatable :: lai_mosaic(:,:,:)
  real, allocatable :: tr_urb2d_mosaic(:,:,:)
  real, allocatable :: tb_urb2d_mosaic(:,:,:)
  real, allocatable :: tg_urb2d_mosaic(:,:,:)
  real, allocatable :: tc_urb2d_mosaic(:,:,:)
  real, allocatable :: qc_urb2d_mosaic(:,:,:)
  real, allocatable :: uc_urb2d_mosaic(:,:,:)
  real, allocatable :: sh_urb2d_mosaic(:,:,:)
  real, allocatable :: lh_urb2d_mosaic(:,:,:)
  real, allocatable :: g_urb2d_mosaic(:,:,:)
  real, allocatable :: rn_urb2d_mosaic(:,:,:)
  real, allocatable :: ts_urb2d_mosaic(:,:,:)
  real, allocatable :: ts_rul_urb2d_mosaic(:,:,:)
  real, allocatable :: tslb_mosaic(:,:,:)
  real, allocatable :: smois_mosaic(:,:,:)
  real, allocatable :: sh2o_mosaic(:,:,:)
  real, allocatable :: trl_urb3d_mosaic(:,:,:)
  real, allocatable :: tbl_urb3d_mosaic(:,:,:)
  real, allocatable :: tgl_urb3d_mosaic(:,:,:)
  integer :: mc_original, increment_t, increment_i
  real :: before_qgh(ncase,1),after_qgh(ncase,1)
  real :: before_glw(ncase,1),after_glw(ncase,1)
  real :: before_swdown(ncase,1),after_swdown(ncase,1)
  real :: before_rainbl(ncase,1),after_rainbl(ncase,1)
  real :: before_sr(ncase,1),after_sr(ncase,1)
  real :: before_chs(ncase,1),after_chs(ncase,1)
  real :: before_cqs2(ncase,1),after_cqs2(ncase,1)
  real :: before_chs2(ncase,1),after_chs2(ncase,1)
  real :: before_rib(ncase,1),after_rib(ncase,1)
  real :: before_vegfra(ncase,1),after_vegfra(ncase,1)
  real :: before_shdmin(ncase,1),after_shdmin(ncase,1)
  real :: before_shdmax(ncase,1),after_shdmax(ncase,1)
  real :: before_tmn(ncase,1),after_tmn(ncase,1)
  real :: before_xland(ncase,1),after_xland(ncase,1)
  real :: before_xice(ncase,1),after_xice(ncase,1)
  real :: before_snoalb(ncase,1),after_snoalb(ncase,1)
  real :: before_embck(ncase,1),after_embck(ncase,1)
  real :: before_tsk(ncase,1),after_tsk(ncase,1)
  real :: before_hfx(ncase,1),after_hfx(ncase,1)
  real :: before_qfx(ncase,1),after_qfx(ncase,1)
  real :: before_lh(ncase,1),after_lh(ncase,1)
  real :: before_grdflx(ncase,1),after_grdflx(ncase,1)
  real :: before_qsfc(ncase,1),after_qsfc(ncase,1)
  real :: before_canwat(ncase,1),after_canwat(ncase,1)
  real :: before_snow(ncase,1),after_snow(ncase,1)
  real :: before_snowc(ncase,1),after_snowc(ncase,1)
  real :: before_snowh(ncase,1),after_snowh(ncase,1)
  real :: before_albedo(ncase,1),after_albedo(ncase,1)
  real :: before_albbck(ncase,1),after_albbck(ncase,1)
  real :: before_emiss(ncase,1),after_emiss(ncase,1)
  real :: before_znt(ncase,1),after_znt(ncase,1)
  real :: before_z0(ncase,1),after_z0(ncase,1)
  real :: before_snotime(ncase,1),after_snotime(ncase,1)
  real :: before_lai(ncase,1),after_lai(ncase,1)
  real :: before_smstav(ncase,1),after_smstav(ncase,1)
  real :: before_smstot(ncase,1),after_smstot(ncase,1)
  real :: before_sfcrunoff(ncase,1),after_sfcrunoff(ncase,1)
  real :: before_udrunoff(ncase,1),after_udrunoff(ncase,1)
  real :: before_acsnow(ncase,1),after_acsnow(ncase,1)
  real :: before_acsnom(ncase,1),after_acsnom(ncase,1)
  real :: before_snopcx(ncase,1),after_snopcx(ncase,1)
  real :: before_potevp(ncase,1),after_potevp(ncase,1)
  real :: before_noahres(ncase,1),after_noahres(ncase,1)
  real :: before_chklowq(ncase,1),after_chklowq(ncase,1)
  real :: before_smois(ncase,4,1),after_smois(ncase,4,1)
  real :: before_tslb(ncase,4,1),after_tslb(ncase,4,1)
  real :: before_sh2o(ncase,4,1),after_sh2o(ncase,4,1)
  real :: before_smcrel(ncase,4,1),after_smcrel(ncase,4,1)
  integer :: before_ivgtyp(ncase,1),after_ivgtyp(ncase,1)
  real, allocatable :: before_tsk_mosaic(:,:,:),after_tsk_mosaic(:,:,:)
  real, allocatable :: before_qsfc_mosaic(:,:,:),after_qsfc_mosaic(:,:,:)
  real, allocatable :: before_canwat_mosaic(:,:,:),after_canwat_mosaic(:,:,:)
  real, allocatable :: before_snow_mosaic(:,:,:),after_snow_mosaic(:,:,:)
  real, allocatable :: before_snowh_mosaic(:,:,:),after_snowh_mosaic(:,:,:)
  real, allocatable :: before_snowc_mosaic(:,:,:),after_snowc_mosaic(:,:,:)
  real, allocatable :: before_albedo_mosaic(:,:,:),after_albedo_mosaic(:,:,:)
  real, allocatable :: before_albbck_mosaic(:,:,:),after_albbck_mosaic(:,:,:)
  real, allocatable :: before_emiss_mosaic(:,:,:),after_emiss_mosaic(:,:,:)
  real, allocatable :: before_embck_mosaic(:,:,:),after_embck_mosaic(:,:,:)
  real, allocatable :: before_znt_mosaic(:,:,:),after_znt_mosaic(:,:,:)
  real, allocatable :: before_z0_mosaic(:,:,:),after_z0_mosaic(:,:,:)
  real, allocatable :: before_hfx_mosaic(:,:,:),after_hfx_mosaic(:,:,:)
  real, allocatable :: before_qfx_mosaic(:,:,:),after_qfx_mosaic(:,:,:)
  real, allocatable :: before_lh_mosaic(:,:,:),after_lh_mosaic(:,:,:)
  real, allocatable :: before_grdflx_mosaic(:,:,:),after_grdflx_mosaic(:,:,:)
  real, allocatable :: before_snotime_mosaic(:,:,:),after_snotime_mosaic(:,:,:)
  real, allocatable :: before_tslb_mosaic(:,:,:),after_tslb_mosaic(:,:,:)
  real, allocatable :: before_smois_mosaic(:,:,:),after_smois_mosaic(:,:,:)
  real, allocatable :: before_sh2o_mosaic(:,:,:),after_sh2o_mosaic(:,:,:)
  integer, allocatable :: before_index(:,:,:)
  real, allocatable :: before_fractions(:,:,:),before_landusef(:,:,:)
  real, allocatable :: increment_sfcrunoff(:,:,:)
  real, allocatable :: increment_udrunoff(:,:,:)
  real, allocatable :: increment_potevp(:,:,:)
  real, allocatable :: increment_acsnom(:,:,:)
  real, allocatable :: increment_snopcx(:,:,:)
  real, allocatable :: increment_acsnow(:,:,:)
  real, allocatable :: urban_pre_tr_urb2d(:,:),urban_post_tr_urb2d(:,:)
  real, allocatable :: urban_pre_tb_urb2d(:,:),urban_post_tb_urb2d(:,:)
  real, allocatable :: urban_pre_tg_urb2d(:,:),urban_post_tg_urb2d(:,:)
  real, allocatable :: urban_pre_tc_urb2d(:,:),urban_post_tc_urb2d(:,:)
  real, allocatable :: urban_pre_qc_urb2d(:,:),urban_post_qc_urb2d(:,:)
  real, allocatable :: urban_pre_uc_urb2d(:,:),urban_post_uc_urb2d(:,:)
  real, allocatable :: urban_pre_xxxr_urb2d(:,:),urban_post_xxxr_urb2d(:,:)
  real, allocatable :: urban_pre_xxxb_urb2d(:,:),urban_post_xxxb_urb2d(:,:)
  real, allocatable :: urban_pre_xxxg_urb2d(:,:),urban_post_xxxg_urb2d(:,:)
  real, allocatable :: urban_pre_xxxc_urb2d(:,:),urban_post_xxxc_urb2d(:,:)
  real, allocatable :: urban_pre_drelr_urb2d(:,:),urban_post_drelr_urb2d(:,:)
  real, allocatable :: urban_pre_drelb_urb2d(:,:),urban_post_drelb_urb2d(:,:)
  real, allocatable :: urban_pre_drelg_urb2d(:,:),urban_post_drelg_urb2d(:,:)
  real, allocatable :: urban_pre_flxhumr_urb2d(:,:),urban_post_flxhumr_urb2d(:,:)
  real, allocatable :: urban_pre_flxhumb_urb2d(:,:),urban_post_flxhumb_urb2d(:,:)
  real, allocatable :: urban_pre_flxhumg_urb2d(:,:),urban_post_flxhumg_urb2d(:,:)
  real, allocatable :: urban_pre_cmcr_urb2d(:,:),urban_post_cmcr_urb2d(:,:)
  real, allocatable :: urban_pre_tgr_urb2d(:,:),urban_post_tgr_urb2d(:,:)
  real, allocatable :: urban_pre_sh_urb2d(:,:),urban_post_sh_urb2d(:,:)
  real, allocatable :: urban_pre_lh_urb2d(:,:),urban_post_lh_urb2d(:,:)
  real, allocatable :: urban_pre_g_urb2d(:,:),urban_post_g_urb2d(:,:)
  real, allocatable :: urban_pre_rn_urb2d(:,:),urban_post_rn_urb2d(:,:)
  real, allocatable :: urban_pre_ts_urb2d(:,:),urban_post_ts_urb2d(:,:)
  real, allocatable :: urban_pre_psim_urb2d(:,:),urban_post_psim_urb2d(:,:)
  real, allocatable :: urban_pre_psih_urb2d(:,:),urban_post_psih_urb2d(:,:)
  real, allocatable :: urban_pre_gz1oz0_urb2d(:,:),urban_post_gz1oz0_urb2d(:,:)
  real, allocatable :: urban_pre_akms_urb2d(:,:),urban_post_akms_urb2d(:,:)
  real, allocatable :: urban_pre_u10_urb2d(:,:),urban_post_u10_urb2d(:,:)
  real, allocatable :: urban_pre_v10_urb2d(:,:),urban_post_v10_urb2d(:,:)
  real, allocatable :: urban_pre_th2_urb2d(:,:),urban_post_th2_urb2d(:,:)
  real, allocatable :: urban_pre_q2_urb2d(:,:),urban_post_q2_urb2d(:,:)
  real, allocatable :: urban_pre_trl_urb3d(:,:,:),urban_post_trl_urb3d(:,:,:)
  real, allocatable :: urban_pre_tbl_urb3d(:,:,:),urban_post_tbl_urb3d(:,:,:)
  real, allocatable :: urban_pre_tgl_urb3d(:,:,:),urban_post_tgl_urb3d(:,:,:)
  real, allocatable :: urban_pre_tgrl_urb3d(:,:,:),urban_post_tgrl_urb3d(:,:,:)
  real, allocatable :: urban_pre_smr_urb3d(:,:,:),urban_post_smr_urb3d(:,:,:)
  real, allocatable :: urban_pre_cosz_urb2d(:,:),urban_post_cosz_urb2d(:,:)
  real, allocatable :: urban_pre_omg_urb2d(:,:),urban_post_omg_urb2d(:,:)
  real, allocatable :: urban_pre_xlat_urb2d(:,:),urban_post_xlat_urb2d(:,:)
  real, allocatable :: urban_pre_cmr_sfcdif(:,:),urban_post_cmr_sfcdif(:,:)
  real, allocatable :: urban_pre_chr_sfcdif(:,:),urban_post_chr_sfcdif(:,:)
  real, allocatable :: urban_pre_cmc_sfcdif(:,:),urban_post_cmc_sfcdif(:,:)
  real, allocatable :: urban_pre_chc_sfcdif(:,:),urban_post_chc_sfcdif(:,:)
  real, allocatable :: urban_pre_cmgr_sfcdif(:,:),urban_post_cmgr_sfcdif(:,:)
  real, allocatable :: urban_pre_chgr_sfcdif(:,:),urban_post_chgr_sfcdif(:,:)
  real, allocatable :: urban_pre_tr_urb2d_mosaic(:,:,:),urban_post_tr_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_tb_urb2d_mosaic(:,:,:),urban_post_tb_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_tg_urb2d_mosaic(:,:,:),urban_post_tg_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_tc_urb2d_mosaic(:,:,:),urban_post_tc_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_qc_urb2d_mosaic(:,:,:),urban_post_qc_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_uc_urb2d_mosaic(:,:,:),urban_post_uc_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_sh_urb2d_mosaic(:,:,:),urban_post_sh_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_lh_urb2d_mosaic(:,:,:),urban_post_lh_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_g_urb2d_mosaic(:,:,:),urban_post_g_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_rn_urb2d_mosaic(:,:,:),urban_post_rn_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_ts_urb2d_mosaic(:,:,:),urban_post_ts_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_ts_rul_urb2d_mosaic(:,:,:),urban_post_ts_rul_urb2d_mosaic(:,:,:)
  real, allocatable :: urban_pre_trl_urb3d_mosaic(:,:,:),urban_post_trl_urb3d_mosaic(:,:,:)
  real, allocatable :: urban_pre_tbl_urb3d_mosaic(:,:,:),urban_post_tbl_urb3d_mosaic(:,:,:)
  real, allocatable :: urban_pre_tgl_urb3d_mosaic(:,:,:),urban_post_tgl_urb3d_mosaic(:,:,:)
  real, allocatable :: urban_pre_ust(:,:),urban_post_ust(:,:)
  real, allocatable :: urban_pre_ust_urb2d(:,:),urban_post_ust_urb2d(:,:)
  real, allocatable :: urban_pre_frc_urb2d(:,:),urban_post_frc_urb2d(:,:)
  call oracle_root(outdir)
  call soil_veg_gen_parm('MODIFIED_IGBP_MODIS_NOAH','STAS')
  subn=transfer(1,1.0);nzero=sign(0.,-1.)
  dzs=[.10,.30,.60,1.00]
  sf_urban_physics=0;lcz=0;nurbm=3
  call get_command_argument(2,arg)
  if(len_trim(arg)==0)then
    call get_command_argument(0,exe)
    call get_command_argument(1,outdir)
    do switch=1,4
      write(arg,'(i0)')switch
      call execute_command_line(trim(exe)//' '//trim(outdir)//' '//trim(arg),exitstat=status)
      if(status/=0)error stop 'UCM child failed'
    enddo
    stop
  endif
  read(arg,*)switch
  wrf_init_frc=switch>=3
  if(wrf_init_frc) switch=switch-2
  use_lcz=switch-1
  call urban_param_init(dzr,dzb,dzg,nsoil,1,use_lcz,.false.)
  ! One already-graded switch set per family, with green roof active in LCZ.
  ch_scheme_data=1;ts_scheme_data=1;ahoption=1;alhoption=1
  imp_scheme=2;iri_scheme=0;groption=use_lcz
  sf_urban_physics=1
  do twin=1,1
  u10=-11.;v10=-12.;psim=-13.;psih=-14.;gz1oz0=-15.;akhs=-16.;akms=-17.
  cmr_sfcdif=0.;chr_sfcdif=0.;cmc_sfcdif=0.;chc_sfcdif=0.
  cmgr_sfcdif=0.;chgr_sfcdif=0.
  mc=3;nlcat=21
  if(switch==5) mc=1
  if(switch==6) mc=5
  if(use_lcz==1) nlcat=61
  usemonalb=switch==2;rdlai2d=switch==2;frpcpn=switch==4
  opt_thcnd=1
  if(switch==3) opt_thcnd=2
  dz8w=0
  qv3d=0
  p8w3d=0
  t3d=0
  tsk=0
  hfx=0
  qfx=0
  lh=0
  grdflx=0
  qgh=0
  gsw=0
  swdown=0
  swddir=0
  swddif=0
  glw=0
  smstav=0
  smstot=0
  sfcrunoff=0
  udrunoff=0
  vegfra=0
  albedo=0
  albbck=0
  znt=0
  z0=0
  tmn=0
  xland=0
  xice=0
  emiss=0
  embck=0
  snowc=0
  qsfc=0
  rainbl=0
  snow=0
  canwat=0
  chs=0
  chs2=0
  cqs2=0
  cpm=0
  sr=0
  chklowq=0
  lai=0
  qz0=0
  snowh=0
  snoalb=0
  shdmin=0
  shdmax=0
  snotime=0
  acsnom=0
  acsnow=0
  snopcx=0
  potevp=0
  rib=0
  noahres=0
  flx4_2d=0
  fvb_2d=0
  fbur_2d=0
  fgsn_2d=0
  ust_urb2d=0
  frc_urb2d=0
  sda_hfx=0
  sda_qfx=0
  hfx_both=0
  qfx_both=0
  qnorm=0
  smois=0
  tslb=0
  sh2o=0
  smcrel=0
  ivgtyp=0
  isltyp=0
  utype_urb2d=0
  dzs=0
  tr_urb2d=0
  tb_urb2d=0
  tg_urb2d=0
  tc_urb2d=0
  qc_urb2d=0
  xxxr_urb2d=0
  xxxb_urb2d=0
  xxxg_urb2d=0
  xxxc_urb2d=0
  drelr_urb2d=0
  drelb_urb2d=0
  drelg_urb2d=0
  flxhumr_urb2d=0
  flxhumb_urb2d=0
  flxhumg_urb2d=0
  cmcr_urb2d=0
  tgr_urb2d=0
  trl_urb3d=0
  tbl_urb3d=0
  tgl_urb3d=0
  tgrl_urb3d=0
  smr_urb3d=0
  sh_urb2d=0
  lh_urb2d=0
  g_urb2d=0
  rn_urb2d=0
  ts_urb2d=0
  trb_urb4d=0
  tw1_urb4d=0
  tw2_urb4d=0
  tgb_urb4d=0
  tlev_urb3d=0
  qlev_urb3d=0
  tw1lev_urb3d=0
  tw2lev_urb3d=0
  tglev_urb3d=0
  tflev_urb3d=0
  lf_ac_urb3d=0
  sf_ac_urb3d=0
  cm_ac_urb3d=0
  sfvent_urb3d=0
  lfvent_urb3d=0
  sfwin1_urb3d=0
  sfwin2_urb3d=0
  sfw1_urb3d=0
  sfw2_urb3d=0
  sfr_urb3d=0
  sfg_urb3d=0
  ep_pv_urb3d=0
  t_pv_urb3d=0
  trv_urb4d=0
  qr_urb4d=0
  qgr_urb3d=0
  tgr_urb3d=0
  drain_urb4d=0
  draingr_urb3d=0
  sfrv_urb3d=0
  lfrv_urb3d=0
  dgr_urb3d=0
  dg_urb3d=0
  lfr_urb3d=0
  lfg_urb3d=0
  hi_urb2d=0
  lp_urb2d=0
  lb_urb2d=0
  hgt_urb2d=0
  mh_urb2d=0
  stdh_urb2d=0
  lf_urb2d=0
  a_u_bep=0
  a_v_bep=0
  a_t_bep=0
  a_q_bep=0
  a_e_bep=0
  b_u_bep=0
  b_v_bep=0
  b_t_bep=0
  b_q_bep=0
  b_e_bep=0
  vl_bep=0
  dlg_bep=0
  sf_bep=0
  dl_u_bep=0
  uc_urb2d=0
  psim_urb2d=0
  psih_urb2d=0
  u10_urb2d=0
  v10_urb2d=0
  gz1oz0_urb2d=0
  akms_urb2d=0
  th2_urb2d=0
  q2_urb2d=0
  cosz_urb2d=0
  omg_urb2d=0
  xlat_urb2d=0
  lf_urb2d_s=0
  z0_urb2d=0
  ust=0
  tsk_rural_bep=0
  xlong=0
  xlat=0
  u_phy=0
  v_phy=0
  th_phy=0
  rho=0
  p_phy=0
  rc2=0.;xlai2=0.
  dzs=[.10,.30,.60,1.00]
  call build_fixture()
  if(switch/=11) then
    rainbl(17,1)=0.;snow(18,1)=0.;snowc(18,1)=0.
    qv3d(19,1,1)=.008;canwat(24,1)=.3
    chs(25,1)=.02;chs2(25,1)=.03;cqs2(25,1)=.03
  endif

  allocate(landusef(ncase,nlcat,1),landusef2(ncase,nlcat,1),mosaic_cat_index(ncase,nlcat,1))
  allocate(tsk_mosaic(ncase,mc,1)); tsk_mosaic=0.
  allocate(qsfc_mosaic(ncase,mc,1)); qsfc_mosaic=0.
  allocate(canwat_mosaic(ncase,mc,1)); canwat_mosaic=0.
  allocate(snow_mosaic(ncase,mc,1)); snow_mosaic=0.
  allocate(snowh_mosaic(ncase,mc,1)); snowh_mosaic=0.
  allocate(snowc_mosaic(ncase,mc,1)); snowc_mosaic=0.
  allocate(albedo_mosaic(ncase,mc,1)); albedo_mosaic=0.
  allocate(albbck_mosaic(ncase,mc,1)); albbck_mosaic=0.
  allocate(emiss_mosaic(ncase,mc,1)); emiss_mosaic=0.
  allocate(embck_mosaic(ncase,mc,1)); embck_mosaic=0.
  allocate(znt_mosaic(ncase,mc,1)); znt_mosaic=0.
  allocate(z0_mosaic(ncase,mc,1)); z0_mosaic=0.
  allocate(hfx_mosaic(ncase,mc,1)); hfx_mosaic=0.
  allocate(qfx_mosaic(ncase,mc,1)); qfx_mosaic=0.
  allocate(lh_mosaic(ncase,mc,1)); lh_mosaic=0.
  allocate(grdflx_mosaic(ncase,mc,1)); grdflx_mosaic=0.
  allocate(snotime_mosaic(ncase,mc,1)); snotime_mosaic=0.
  allocate(rc_mosaic(ncase,mc,1)); rc_mosaic=0.
  allocate(lai_mosaic(ncase,mc,1)); lai_mosaic=0.
  allocate(tr_urb2d_mosaic(ncase,mc,1)); tr_urb2d_mosaic=0.
  allocate(tb_urb2d_mosaic(ncase,mc,1)); tb_urb2d_mosaic=0.
  allocate(tg_urb2d_mosaic(ncase,mc,1)); tg_urb2d_mosaic=0.
  allocate(tc_urb2d_mosaic(ncase,mc,1)); tc_urb2d_mosaic=0.
  allocate(qc_urb2d_mosaic(ncase,mc,1)); qc_urb2d_mosaic=0.
  allocate(uc_urb2d_mosaic(ncase,mc,1)); uc_urb2d_mosaic=0.
  allocate(sh_urb2d_mosaic(ncase,mc,1)); sh_urb2d_mosaic=0.
  allocate(lh_urb2d_mosaic(ncase,mc,1)); lh_urb2d_mosaic=0.
  allocate(g_urb2d_mosaic(ncase,mc,1)); g_urb2d_mosaic=0.
  allocate(rn_urb2d_mosaic(ncase,mc,1)); rn_urb2d_mosaic=0.
  allocate(ts_urb2d_mosaic(ncase,mc,1)); ts_urb2d_mosaic=0.
  allocate(ts_rul_urb2d_mosaic(ncase,mc,1)); ts_rul_urb2d_mosaic=0.
  allocate(tslb_mosaic(ncase,4*mc,1)); tslb_mosaic=0.
  allocate(smois_mosaic(ncase,4*mc,1)); smois_mosaic=0.
  allocate(sh2o_mosaic(ncase,4*mc,1)); sh2o_mosaic=0.
  allocate(trl_urb3d_mosaic(ncase,4*mc,1)); trl_urb3d_mosaic=0.
  allocate(tbl_urb3d_mosaic(ncase,4*mc,1)); tbl_urb3d_mosaic=0.
  allocate(tgl_urb3d_mosaic(ncase,4*mc,1)); tgl_urb3d_mosaic=0.
  landusef=0.
  xland=1.;xice=0.;snow=0.;snowc=0.;snowh=0.;canwat=.3
  qv3d=.008;t3d=290.;dz8w=160.;qgh=.012;chs=.005;chs2=.007;cqs2=.008
  tsk=291.;tslb=288.;smois=.28;sh2o=.28;glw=330.;swdown=500.
  do i=1,ncase
    landusef(i,7,1)=.6;landusef(i,10,1)=.25;landusef(i,12,1)=.15
    select case(mod(i-1,4))
    case(0)
      landusef(i,7,1)=0.;landusef(i,13,1)=.6
    case(1)
      landusef(i,10,1)=0.;landusef(i,13,1)=.25
    case(2)
      landusef(i,12,1)=0.;landusef(i,13,1)=.15
    end select
    ivgtyp(i,1)=merge(13,7,mod(i-1,4)==0)
    if(use_lcz==1.and.mod(i-1,4)<2)then
      landusef(i,:,1)=0.;landusef(i,51,1)=.6;landusef(i,54,1)=.25;landusef(i,7,1)=.15
      ivgtyp(i,1)=51
    endif
  enddo
    call urban_var_init(ISURBAN=13, TSURFACE0_URB=tsk, &
      TLAYER0_URB=tslb, TDEEP0_URB=tmn, IVGTYP=ivgtyp, &
      ims=1, ime=ncase, jms=1, jme=1, kms=1, kme=nlev, &
      num_soil_layers=nsoil, &
      LCZ_1=51, LCZ_2=52, LCZ_3=53, LCZ_4=54, &
      LCZ_5=55, LCZ_6=56, LCZ_7=57, LCZ_8=58, &
      LCZ_9=59, LCZ_10=60, LCZ_11=61, &
      restart=.false., sf_urban_physics=sf_urban_physics, &
      XXXR_URB2D=xxxr_urb2d, XXXB_URB2D=xxxb_urb2d, XXXG_URB2D=xxxg_urb2d, XXXC_URB2D=xxxc_urb2d, &
      TR_URB2D=tr_urb2d, TB_URB2D=tb_urb2d, TG_URB2D=tg_urb2d, TC_URB2D=tc_urb2d, QC_URB2D=qc_urb2d, &
      TRL_URB3D=trl_urb3d, TBL_URB3D=tbl_urb3d, TGL_URB3D=tgl_urb3d, &
      SH_URB2D=sh_urb2d, LH_URB2D=lh_urb2d, G_URB2D=g_urb2d, RN_URB2D=rn_urb2d, TS_URB2D=ts_urb2d, &
      num_urban_ndm=num_urban_ndm, urban_map_zrd=urban_map_zrd, urban_map_zwd=urban_map_zwd, &
      urban_map_gd=urban_map_gd, urban_map_zd=urban_map_zd, urban_map_zdf=urban_map_zdf, &
      urban_map_bd=urban_map_bd, urban_map_wd=urban_map_wd, urban_map_gbd=urban_map_gbd, &
      urban_map_fbd=urban_map_fbd, urban_map_zgrd=urban_map_zgrd, num_urban_hi=num_urban_hi, &
      TRB_URB4D=trb_urb4d, TW1_URB4D=tw1_urb4d, TW2_URB4D=tw2_urb4d, TGB_URB4D=tgb_urb4d, &
      TLEV_URB3D=tlev_urb3d, QLEV_URB3D=qlev_urb3d, TW1LEV_URB3D=tw1lev_urb3d, &
      TW2LEV_URB3D=tw2lev_urb3d, TGLEV_URB3D=tglev_urb3d, TFLEV_URB3D=tflev_urb3d, &
      SF_AC_URB3D=sf_ac_urb3d, LF_AC_URB3D=lf_ac_urb3d, CM_AC_URB3D=cm_ac_urb3d, &
      SFVENT_URB3D=sfvent_urb3d, LFVENT_URB3D=lfvent_urb3d, &
      SFWIN1_URB3D=sfwin1_urb3d, SFWIN2_URB3D=sfwin2_urb3d, &
      SFW1_URB3D=sfw1_urb3d, SFW2_URB3D=sfw2_urb3d, SFR_URB3D=sfr_urb3d, SFG_URB3D=sfg_urb3d, &
      EP_PV_URB3D=ep_pv_urb3d, T_PV_URB3D=t_pv_urb3d, &
      TRV_URB4D=trv_urb4d, QR_URB4D=qr_urb4d, QGR_URB3D=qgr_urb3d, TGR_URB3D=tgr_urb3d, &
      DRAIN_URB4D=drain_urb4d, DRAINGR_URB3D=draingr_urb3d, SFRV_URB3D=sfrv_urb3d, &
      LFRV_URB3D=lfrv_urb3d, DGR_URB3D=dgr_urb3d, DG_URB3D=dg_urb3d, LFR_URB3D=lfr_urb3d, &
      LFG_URB3D=lfg_urb3d, SMOIS_URB=smois, &
      LP_URB2D=lp_urb2d, HI_URB2D=hi_urb2d, LB_URB2D=lb_urb2d, HGT_URB2D=hgt_urb2d, MH_URB2D=mh_urb2d, &
      STDH_URB2D=stdh_urb2d, LF_URB2D=lf_urb2d, &
      CMCR_URB2D=cmcr_urb2d, TGR_URB2D=tgr_urb2d, TGRL_URB3D=tgrl_urb3d, SMR_URB3D=smr_urb3d, &
      DRELR_URB2D=drelr_urb2d, DRELB_URB2D=drelb_urb2d, DRELG_URB2D=drelg_urb2d, &
      FLXHUMR_URB2D=flxhumr_urb2d, FLXHUMB_URB2D=flxhumb_urb2d, &
      FLXHUMG_URB2D=flxhumg_urb2d, &
      A_U_BEP=a_u_bep, A_V_BEP=a_v_bep, A_T_BEP=a_t_bep, A_Q_BEP=a_q_bep, A_E_BEP=a_e_bep, &
      B_U_BEP=b_u_bep, B_V_BEP=b_v_bep, B_T_BEP=b_t_bep, B_Q_BEP=b_q_bep, B_E_BEP=b_e_bep, &
      DLG_BEP=dlg_bep, DL_U_BEP=dl_u_bep, SF_BEP=sf_bep, VL_BEP=vl_bep, &
      FRC_URB2D=frc_urb2d, UTYPE_URB2D=utype_urb2d, USE_WUDAPT_LCZ=use_lcz)
  call lsm_mosaic_init(ivgtyp,17,13,15,xland,xice,0, &
    tsk,tslb,smois,sh2o,snow,snowc,snowh,canwat, &
    1,ncase+1,1,2,1,2,1,ncase,1,1,1,2,1,ncase,1,1,1,1,.false., &
    landusef,landusef2,nlcat,4,1,mc,mosaic_cat_index, &
    tsk_mosaic,tslb_mosaic,smois_mosaic,sh2o_mosaic, &
    canwat_mosaic,snow_mosaic,snowh_mosaic,snowc_mosaic, &
    albedo,albbck,emiss,embck,znt, &
    albedo_mosaic,albbck_mosaic,emiss_mosaic,embck_mosaic,znt_mosaic,z0_mosaic, &
    tr_urb2d_mosaic,tb_urb2d_mosaic,tg_urb2d_mosaic,tc_urb2d_mosaic,qc_urb2d_mosaic, &
    trl_urb3d_mosaic,tbl_urb3d_mosaic,tgl_urb3d_mosaic, &
    sh_urb2d_mosaic,lh_urb2d_mosaic,g_urb2d_mosaic,rn_urb2d_mosaic, &
    ts_urb2d_mosaic,ts_rul_urb2d_mosaic)
  do i=1,ncase
    if(.not.wrf_init_frc)then
    select case(mod((i-1)/4,5))
    case(0);frc_urb2d(i,1)=.1
    case(1);frc_urb2d(i,1)=.5
    case(2);frc_urb2d(i,1)=.95
    case(3);frc_urb2d(i,1)=.99
    case(4);frc_urb2d(i,1)=1.
    end select
    endif
    u_phy(i,:,1)=.2+mod(i,5);v_phy(i,:,1)=.3
    omg_urb2d(i,1)=.1*i;xlat_urb2d(i,1)=45.+.1*i;cosz_urb2d(i,1)=.7
    xxxr_urb2d(i,1)=.01;xxxb_urb2d(i,1)=.02;xxxg_urb2d(i,1)=.03;xxxc_urb2d(i,1)=.04
    tgr_urb2d(i,1)=289.+.1*i;cmcr_urb2d(i,1)=.001
    tgrl_urb3d(i,:,1)=287.+.1*i;smr_urb3d(i,:,1)=.25+.001*i
    drelr_urb2d(i,1)=.01;drelb_urb2d(i,1)=.02;drelg_urb2d(i,1)=.03
    flxhumr_urb2d(i,1)=.001;flxhumb_urb2d(i,1)=.002;flxhumg_urb2d(i,1)=.003
  enddo
  do t=1,mc
    tr_urb2d_mosaic(:,t,1)=293.+.3*t
    tb_urb2d_mosaic(:,t,1)=292.+.2*t
    tg_urb2d_mosaic(:,t,1)=291.+.1*t
    tc_urb2d_mosaic(:,t,1)=290.+.4*t
    qc_urb2d_mosaic(:,t,1)=.007+.001*t
    uc_urb2d_mosaic(:,t,1)=.2*t
    ts_urb2d_mosaic(:,t,1)=294.+.2*t
    ts_rul_urb2d_mosaic(:,t,1)=287.+.7*t
    do k=1,4
      trl_urb3d_mosaic(:,4*(t-1)+k,1)=289.+.2*t+.1*k
      tbl_urb3d_mosaic(:,4*(t-1)+k,1)=288.+.3*t+.1*k
      tgl_urb3d_mosaic(:,4*(t-1)+k,1)=287.+.4*t+.1*k
    enddo
    tsk_mosaic(:,t,1)=tsk(:,1)+.3*(t-1)
    qsfc_mosaic(:,t,1)=qsfc(:,1)+.001*(t-2)
    canwat_mosaic(:,t,1)=canwat(:,1)*(.8+.1*t)
    snow_mosaic(:,t,1)=snow(:,1)*(.8+.1*t)
    snowh_mosaic(:,t,1)=snowh(:,1)*(.8+.1*t)
    snotime_mosaic(:,t,1)=snotime(:,1)+60.*t
    albedo_mosaic(:,t,1)=albedo(:,1)+.005*t
    albbck_mosaic(:,t,1)=albbck(:,1)+.005*t
    emiss_mosaic(:,t,1)=emiss(:,1)-.001*t
    embck_mosaic(:,t,1)=embck(:,1)-.001*t
    znt_mosaic(:,t,1)=znt(:,1)+.001*t
    z0_mosaic(:,t,1)=z0(:,1)+.001*t
    do ns=1,4
      tslb_mosaic(:,4*(t-1)+ns,1)=tslb(:,ns,1)+.2*(t-1)
      smois_mosaic(:,4*(t-1)+ns,1)=smois(:,ns,1)+.001*(t-1)
      sh2o_mosaic(:,4*(t-1)+ns,1)=sh2o(:,ns,1)+.001*(t-1)
    enddo
  enddo
  do step=1,4
    itimestep=step
    if(step==2) then
      swdown=0.;gsw=0.;rainbl=2.;qv3d=.002
    elseif(step==3) then
      rainbl=0.;sr=0.;t3d=288.;qv3d=.002;glw=300.
    elseif(step==4) then
      rainbl=0.;sr=0.;swdown=500.;gsw=400.
    endif
    write(case_name,'(a,i0,a,i0)') 'v',switch,'_s',step
    if(wrf_init_frc.and.use_lcz==0)then
      call oracle_open('ucm_wrfinit/'//trim(case_name))
    elseif(wrf_init_frc)then
      call oracle_open('ucm_lcz_wrfinit/'//trim(case_name))
    elseif(use_lcz==0)then
      call oracle_open('ucm/'//trim(case_name))
    else
      call oracle_open('ucm_lcz/'//trim(case_name))
    endif
    call oracle_put('use_wudapt_lcz',use_lcz)
    call oracle_put('ucm_table_zr',ZR_TBL)
    call oracle_put('ucm_table_z0c',Z0C_TBL)
    call oracle_put('ucm_table_z0hc',Z0HC_TBL)
    call oracle_put('ucm_table_zdc',ZDC_TBL)
    call oracle_put('ucm_table_svf',SVF_TBL)
    call oracle_put('ucm_table_r',R_TBL)
    call oracle_put('ucm_table_rw',RW_TBL)
    call oracle_put('ucm_table_hgt',HGT_TBL)
    call oracle_put('ucm_table_ah',AH_TBL)
    call oracle_put('ucm_table_alh',ALH_TBL)
    call oracle_put('ucm_table_betr',BETR_TBL)
    call oracle_put('ucm_table_betb',BETB_TBL)
    call oracle_put('ucm_table_betg',BETG_TBL)
    call oracle_put('ucm_table_capr',CAPR_TBL)
    call oracle_put('ucm_table_capb',CAPB_TBL)
    call oracle_put('ucm_table_capg',CAPG_TBL)
    call oracle_put('ucm_table_aksr',AKSR_TBL)
    call oracle_put('ucm_table_aksb',AKSB_TBL)
    call oracle_put('ucm_table_aksg',AKSG_TBL)
    call oracle_put('ucm_table_albr',ALBR_TBL)
    call oracle_put('ucm_table_albb',ALBB_TBL)
    call oracle_put('ucm_table_albg',ALBG_TBL)
    call oracle_put('ucm_table_epsr',EPSR_TBL)
    call oracle_put('ucm_table_epsb',EPSB_TBL)
    call oracle_put('ucm_table_epsg',EPSG_TBL)
    call oracle_put('ucm_table_z0r',Z0R_TBL)
    call oracle_put('ucm_table_z0b',Z0B_TBL)
    call oracle_put('ucm_table_z0g',Z0G_TBL)
    call oracle_put('ucm_table_z0hb',Z0HB_TBL)
    call oracle_put('ucm_table_z0hg',Z0HG_TBL)
    call oracle_put('ucm_table_trlend',TRLEND_TBL)
    call oracle_put('ucm_table_tblend',TBLEND_TBL)
    call oracle_put('ucm_table_tglend',TGLEND_TBL)
    call oracle_put('ucm_table_akanda_urban',AKANDA_URBAN_TBL)
    call oracle_put('ucm_table_frc_urb',frc_urb_tbl)
    call oracle_put('ucm_global_dzr',dzr)
    call oracle_put('ucm_global_dzb',dzb)
    call oracle_put('ucm_global_dzg',dzg)
    call oracle_put('ucm_global_dzgr',dzgr)
    call oracle_put('ucm_global_porimp',porimp)
    call oracle_put('ucm_global_dengimp',dengimp)
    call oracle_put('ucm_global_ahdiuprf',ahdiuprf)
    call oracle_put('ucm_global_alhseason',alhseason)
    call oracle_put('ucm_global_alhdiuprf',alhdiuprf)
    call oracle_put('ucm_global_fgr',fgr)
    call oracle_put('ucm_boundr',boundr_data)
    call oracle_put('ucm_boundb',boundb_data)
    call oracle_put('ucm_boundg',boundg_data)
    call oracle_put('ucm_oasis',oasis)
    call oracle_put('ucm_ch_scheme',ch_scheme_data)
    call oracle_put('ucm_ts_scheme',ts_scheme_data)
    call oracle_put('ucm_ahoption',ahoption)
    call oracle_put('ucm_alhoption',alhoption)
    call oracle_put('ucm_imp_scheme',imp_scheme)
    call oracle_put('ucm_iri_scheme',iri_scheme)
    call oracle_put('ucm_groption',groption)
    call oracle_put('julian',1);call oracle_put('julyr',1974)
    call oracle_put('declin',declin)
    call oracle_put('variant',switch)
    call oracle_put('mosaic_cat',mc);call oracle_put('nlcat',nlcat)
    call oracle_put('dt',dt);call oracle_put('itimestep',itimestep)
    call oracle_put('rdlai2d',merge(1,0,rdlai2d));call oracle_put('usemonalb',merge(1,0,usemonalb))
    call oracle_put('frpcpn',merge(1,0,frpcpn));call oracle_put('opt_thcnd',opt_thcnd)
    mc_original=mc
    before_index=mosaic_cat_index
    before_fractions=landusef2
    before_landusef=landusef
    before_qgh=qgh
    before_glw=glw
    before_swdown=swdown
    before_rainbl=rainbl
    before_sr=sr
    before_chs=chs
    before_cqs2=cqs2
    before_chs2=chs2
    before_rib=rib
    before_vegfra=vegfra
    before_shdmin=shdmin
    before_shdmax=shdmax
    before_tmn=tmn
    before_xland=xland
    before_xice=xice
    before_snoalb=snoalb
    before_embck=embck
    before_tsk=tsk
    before_hfx=hfx
    before_qfx=qfx
    before_lh=lh
    before_grdflx=grdflx
    before_qsfc=qsfc
    before_canwat=canwat
    before_snow=snow
    before_snowc=snowc
    before_snowh=snowh
    before_albedo=albedo
    before_albbck=albbck
    before_emiss=emiss
    before_znt=znt
    before_z0=z0
    before_snotime=snotime
    before_lai=lai
    before_smstav=smstav
    before_smstot=smstot
    before_sfcrunoff=sfcrunoff
    before_udrunoff=udrunoff
    before_acsnow=acsnow
    before_acsnom=acsnom
    before_snopcx=snopcx
    before_potevp=potevp
    before_noahres=noahres
    before_chklowq=chklowq
    before_smois=smois
    before_tslb=tslb
    before_sh2o=sh2o
    before_smcrel=smcrel
    before_ivgtyp=ivgtyp
    before_tsk_mosaic=tsk_mosaic
    before_qsfc_mosaic=qsfc_mosaic
    before_canwat_mosaic=canwat_mosaic
    before_snow_mosaic=snow_mosaic
    before_snowh_mosaic=snowh_mosaic
    before_snowc_mosaic=snowc_mosaic
    before_albedo_mosaic=albedo_mosaic
    before_albbck_mosaic=albbck_mosaic
    before_emiss_mosaic=emiss_mosaic
    before_embck_mosaic=embck_mosaic
    before_znt_mosaic=znt_mosaic
    before_z0_mosaic=z0_mosaic
    before_hfx_mosaic=hfx_mosaic
    before_qfx_mosaic=qfx_mosaic
    before_lh_mosaic=lh_mosaic
    before_grdflx_mosaic=grdflx_mosaic
    before_snotime_mosaic=snotime_mosaic
    before_tslb_mosaic=tslb_mosaic
    before_smois_mosaic=smois_mosaic
    before_sh2o_mosaic=sh2o_mosaic
    urban_pre_tr_urb2d=tr_urb2d
    urban_pre_tb_urb2d=tb_urb2d
    urban_pre_tg_urb2d=tg_urb2d
    urban_pre_tc_urb2d=tc_urb2d
    urban_pre_qc_urb2d=qc_urb2d
    urban_pre_uc_urb2d=uc_urb2d
    urban_pre_xxxr_urb2d=xxxr_urb2d
    urban_pre_xxxb_urb2d=xxxb_urb2d
    urban_pre_xxxg_urb2d=xxxg_urb2d
    urban_pre_xxxc_urb2d=xxxc_urb2d
    urban_pre_drelr_urb2d=drelr_urb2d
    urban_pre_drelb_urb2d=drelb_urb2d
    urban_pre_drelg_urb2d=drelg_urb2d
    urban_pre_flxhumr_urb2d=flxhumr_urb2d
    urban_pre_flxhumb_urb2d=flxhumb_urb2d
    urban_pre_flxhumg_urb2d=flxhumg_urb2d
    urban_pre_cmcr_urb2d=cmcr_urb2d
    urban_pre_tgr_urb2d=tgr_urb2d
    urban_pre_sh_urb2d=sh_urb2d
    urban_pre_lh_urb2d=lh_urb2d
    urban_pre_g_urb2d=g_urb2d
    urban_pre_rn_urb2d=rn_urb2d
    urban_pre_ts_urb2d=ts_urb2d
    urban_pre_psim_urb2d=psim_urb2d
    urban_pre_psih_urb2d=psih_urb2d
    urban_pre_gz1oz0_urb2d=gz1oz0_urb2d
    urban_pre_akms_urb2d=akms_urb2d
    urban_pre_u10_urb2d=u10_urb2d
    urban_pre_v10_urb2d=v10_urb2d
    urban_pre_th2_urb2d=th2_urb2d
    urban_pre_q2_urb2d=q2_urb2d
    urban_pre_trl_urb3d=trl_urb3d
    urban_pre_tbl_urb3d=tbl_urb3d
    urban_pre_tgl_urb3d=tgl_urb3d
    urban_pre_tgrl_urb3d=tgrl_urb3d
    urban_pre_smr_urb3d=smr_urb3d
    urban_pre_cosz_urb2d=cosz_urb2d
    urban_pre_omg_urb2d=omg_urb2d
    urban_pre_xlat_urb2d=xlat_urb2d
    urban_pre_cmr_sfcdif=cmr_sfcdif
    urban_pre_chr_sfcdif=chr_sfcdif
    urban_pre_cmc_sfcdif=cmc_sfcdif
    urban_pre_chc_sfcdif=chc_sfcdif
    urban_pre_cmgr_sfcdif=cmgr_sfcdif
    urban_pre_chgr_sfcdif=chgr_sfcdif
    urban_pre_tr_urb2d_mosaic=tr_urb2d_mosaic
    urban_pre_tb_urb2d_mosaic=tb_urb2d_mosaic
    urban_pre_tg_urb2d_mosaic=tg_urb2d_mosaic
    urban_pre_tc_urb2d_mosaic=tc_urb2d_mosaic
    urban_pre_qc_urb2d_mosaic=qc_urb2d_mosaic
    urban_pre_uc_urb2d_mosaic=uc_urb2d_mosaic
    urban_pre_sh_urb2d_mosaic=sh_urb2d_mosaic
    urban_pre_lh_urb2d_mosaic=lh_urb2d_mosaic
    urban_pre_g_urb2d_mosaic=g_urb2d_mosaic
    urban_pre_rn_urb2d_mosaic=rn_urb2d_mosaic
    urban_pre_ts_urb2d_mosaic=ts_urb2d_mosaic
    urban_pre_ts_rul_urb2d_mosaic=ts_rul_urb2d_mosaic
    urban_pre_trl_urb3d_mosaic=trl_urb3d_mosaic
    urban_pre_tbl_urb3d_mosaic=tbl_urb3d_mosaic
    urban_pre_tgl_urb3d_mosaic=tgl_urb3d_mosaic
    urban_pre_ust=ust
    urban_pre_ust_urb2d=ust_urb2d
    urban_pre_frc_urb2d=frc_urb2d
    call dump('_in')
  call lsm_mosaic(dz8w, qv3d, p8w3d, t3d, tsk,                              &
           hfx, qfx, lh, grdflx, qgh, gsw, swdown,   &
           glw, smstav, smstot,                                      &
           sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
           albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
           snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
           nsoil, dt, dzs, itimestep,                                &
           smois, tslb, snow, canwat,                                &
           chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
           .false., frpcpn,                                          &
           sh2o, snowh,                                              &
           shdavg = vegfra, snoalb = snoalb, shdmin = shdmin, shdmax = shdmax,        &
           snotime = snotime,                                        &
           acsnom = acsnom, acsnow = acsnow,                         &
           snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
           xice_threshold = 0.5,                                     &
           rdlai2d = rdlai2d, usemonalb = usemonalb,                 &
           rib = rib, noahres = noahres, opt_thcnd = opt_thcnd,      &
           ua_phys = .false., flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
           fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
           ids = 1, ide = ncase+1, jds = 1, jde = 2, kds = 1, kde = nlev, &
           ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = nlev, &
           its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = nlev-1, &
           use_wudapt_lcz=use_lcz, slucm_distributed_drag=.false., sf_urban_physics = sf_urban_physics, ust_urb2d = ust_urb2d,              &
           num_roof_layers = nsoil, num_wall_layers = nsoil,                 &
           num_road_layers = nsoil, julian = 1, julyr = 1974,            &
           frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
           num_urban_ndm = num_urban_ndm, urban_map_zrd = urban_map_zrd, urban_map_zwd = urban_map_zwd,  &
           urban_map_gd = urban_map_gd, urban_map_zd = urban_map_zd, urban_map_zdf = urban_map_zdf,    &
           urban_map_bd = urban_map_bd, urban_map_wd = urban_map_wd, urban_map_gbd = urban_map_gbd,    &
           urban_map_fbd = urban_map_fbd, urban_map_zgrd = urban_map_zgrd, num_urban_hi = num_urban_hi,  &
           tr_urb2d = tr_urb2d, &
           tb_urb2d = tb_urb2d, &
           tg_urb2d = tg_urb2d, &
           tc_urb2d = tc_urb2d, &
           qc_urb2d = qc_urb2d, &
           xxxr_urb2d = xxxr_urb2d, &
           xxxb_urb2d = xxxb_urb2d, &
           xxxg_urb2d = xxxg_urb2d, &
           xxxc_urb2d = xxxc_urb2d, &
           drelr_urb2d = drelr_urb2d, &
           drelb_urb2d = drelb_urb2d, &
           drelg_urb2d = drelg_urb2d, &
           flxhumr_urb2d = flxhumr_urb2d, &
           flxhumb_urb2d = flxhumb_urb2d, &
           flxhumg_urb2d = flxhumg_urb2d, &
           cmcr_urb2d = cmcr_urb2d, &
           tgr_urb2d = tgr_urb2d, &
           trl_urb3d = trl_urb3d, &
           tbl_urb3d = tbl_urb3d, &
           tgl_urb3d = tgl_urb3d, &
           tgrl_urb3d = tgrl_urb3d, &
           smr_urb3d = smr_urb3d, &
           sh_urb2d = sh_urb2d, &
           lh_urb2d = lh_urb2d, &
           g_urb2d = g_urb2d, &
           rn_urb2d = rn_urb2d, &
           ts_urb2d = ts_urb2d, &
           trb_urb4d = trb_urb4d, &
           tw1_urb4d = tw1_urb4d, &
           tw2_urb4d = tw2_urb4d, &
           tgb_urb4d = tgb_urb4d, &
           tlev_urb3d = tlev_urb3d, &
           qlev_urb3d = qlev_urb3d, &
           tw1lev_urb3d = tw1lev_urb3d, &
           tw2lev_urb3d = tw2lev_urb3d, &
           tglev_urb3d = tglev_urb3d, &
           tflev_urb3d = tflev_urb3d, &
           lf_ac_urb3d = lf_ac_urb3d, &
           sf_ac_urb3d = sf_ac_urb3d, &
           cm_ac_urb3d = cm_ac_urb3d, &
           sfvent_urb3d = sfvent_urb3d, &
           lfvent_urb3d = lfvent_urb3d, &
           sfwin1_urb3d = sfwin1_urb3d, &
           sfwin2_urb3d = sfwin2_urb3d, &
           sfw1_urb3d = sfw1_urb3d, &
           sfw2_urb3d = sfw2_urb3d, &
           sfr_urb3d = sfr_urb3d, &
           sfg_urb3d = sfg_urb3d, &
           ep_pv_urb3d = ep_pv_urb3d, &
           t_pv_urb3d = t_pv_urb3d, &
           trv_urb4d = trv_urb4d, &
           qr_urb4d = qr_urb4d, &
           qgr_urb3d = qgr_urb3d, &
           tgr_urb3d = tgr_urb3d, &
           drain_urb4d = drain_urb4d, &
           draingr_urb3d = draingr_urb3d, &
           sfrv_urb3d = sfrv_urb3d, &
           lfrv_urb3d = lfrv_urb3d, &
           dgr_urb3d = dgr_urb3d, &
           dg_urb3d = dg_urb3d, &
           lfr_urb3d = lfr_urb3d, &
           lfg_urb3d = lfg_urb3d, &
           hi_urb2d = hi_urb2d, &
           lp_urb2d = lp_urb2d, &
           lb_urb2d = lb_urb2d, &
           hgt_urb2d = hgt_urb2d, &
           mh_urb2d = mh_urb2d, &
           stdh_urb2d = stdh_urb2d, &
           lf_urb2d = lf_urb2d, &
           a_u_bep = a_u_bep, &
           a_v_bep = a_v_bep, &
           a_t_bep = a_t_bep, &
           a_q_bep = a_q_bep, &
           a_e_bep = a_e_bep, &
           b_u_bep = b_u_bep, &
           b_v_bep = b_v_bep, &
           b_t_bep = b_t_bep, &
           b_q_bep = b_q_bep, &
           b_e_bep = b_e_bep, &
           vl_bep = vl_bep, &
           dlg_bep = dlg_bep, &
           sf_bep = sf_bep, &
           dl_u_bep = dl_u_bep, &
           uc_urb2d = uc_urb2d, &
           psim_urb2d = psim_urb2d, &
           psih_urb2d = psih_urb2d, &
           u10_urb2d = u10_urb2d, &
           v10_urb2d = v10_urb2d, &
           gz1oz0_urb2d = gz1oz0_urb2d, &
           akms_urb2d = akms_urb2d, &
           th2_urb2d = th2_urb2d, &
           q2_urb2d = q2_urb2d, &
           cosz_urb2d = cosz_urb2d, &
           omg_urb2d = omg_urb2d, &
           xlat_urb2d = xlat_urb2d, &
           lf_urb2d_s = lf_urb2d_s, &
           z0_urb2d = z0_urb2d, &
           ust = ust, &
           tsk_rural_bep = tsk_rural_bep, &
           xlong = xlong, &
           xlat = xlat, &
           u_phy = u_phy, &
           v_phy = v_phy, &
           th_phy = th_phy, &
           rho = rho, &
           p_phy = p_phy, &
           cmr_sfcdif=cmr_sfcdif, &
           chr_sfcdif=chr_sfcdif, &
           cmc_sfcdif=cmc_sfcdif, &
           chc_sfcdif=chc_sfcdif, &
           cmgr_sfcdif=cmgr_sfcdif, &
           chgr_sfcdif=chgr_sfcdif, &
           dzr=dzr, dzb=dzb, dzg=dzg, declin_urb=0.2, gmt=12.0, julday=180, &
 nlcat=nlcat, &
 landusef=landusef, &
 landusef2=landusef2, &
 sf_surface_mosaic=1, &
 mosaic_cat=mc, &
 mosaic_cat_index=mosaic_cat_index, &
 tsk_mosaic=tsk_mosaic, &
 qsfc_mosaic=qsfc_mosaic, &
 canwat_mosaic=canwat_mosaic, &
 snow_mosaic=snow_mosaic, &
 snowh_mosaic=snowh_mosaic, &
 snowc_mosaic=snowc_mosaic, &
 albedo_mosaic=albedo_mosaic, &
 albbck_mosaic=albbck_mosaic, &
 emiss_mosaic=emiss_mosaic, &
 embck_mosaic=embck_mosaic, &
 znt_mosaic=znt_mosaic, &
 z0_mosaic=z0_mosaic, &
 hfx_mosaic=hfx_mosaic, &
 qfx_mosaic=qfx_mosaic, &
 lh_mosaic=lh_mosaic, &
 grdflx_mosaic=grdflx_mosaic, &
 snotime_mosaic=snotime_mosaic, &
 rc_mosaic=rc_mosaic, &
 lai_mosaic=lai_mosaic, &
 tr_urb2d_mosaic=tr_urb2d_mosaic, &
 tb_urb2d_mosaic=tb_urb2d_mosaic, &
 tg_urb2d_mosaic=tg_urb2d_mosaic, &
 tc_urb2d_mosaic=tc_urb2d_mosaic, &
 qc_urb2d_mosaic=qc_urb2d_mosaic, &
 uc_urb2d_mosaic=uc_urb2d_mosaic, &
 sh_urb2d_mosaic=sh_urb2d_mosaic, &
 lh_urb2d_mosaic=lh_urb2d_mosaic, &
 g_urb2d_mosaic=g_urb2d_mosaic, &
 rn_urb2d_mosaic=rn_urb2d_mosaic, &
 ts_urb2d_mosaic=ts_urb2d_mosaic, &
 ts_rul2d_mosaic=ts_rul_urb2d_mosaic, &
 tslb_mosaic=tslb_mosaic, &
 smois_mosaic=smois_mosaic, &
 sh2o_mosaic=sh2o_mosaic, &
 trl_urb3d_mosaic=trl_urb3d_mosaic, &
 tbl_urb3d_mosaic=tbl_urb3d_mosaic, &
 tgl_urb3d_mosaic=tgl_urb3d_mosaic, &
 sda_hfx=sda_hfx,sda_qfx=sda_qfx,hfx_both=hfx_both,qfx_both=qfx_both,qnorm=qnorm,fasdas=0,rc2=rc2,xlai2=xlai2)
    urban_post_tr_urb2d=tr_urb2d
    urban_post_tb_urb2d=tb_urb2d
    urban_post_tg_urb2d=tg_urb2d
    urban_post_tc_urb2d=tc_urb2d
    urban_post_qc_urb2d=qc_urb2d
    urban_post_uc_urb2d=uc_urb2d
    urban_post_xxxr_urb2d=xxxr_urb2d
    urban_post_xxxb_urb2d=xxxb_urb2d
    urban_post_xxxg_urb2d=xxxg_urb2d
    urban_post_xxxc_urb2d=xxxc_urb2d
    urban_post_drelr_urb2d=drelr_urb2d
    urban_post_drelb_urb2d=drelb_urb2d
    urban_post_drelg_urb2d=drelg_urb2d
    urban_post_flxhumr_urb2d=flxhumr_urb2d
    urban_post_flxhumb_urb2d=flxhumb_urb2d
    urban_post_flxhumg_urb2d=flxhumg_urb2d
    urban_post_cmcr_urb2d=cmcr_urb2d
    urban_post_tgr_urb2d=tgr_urb2d
    urban_post_sh_urb2d=sh_urb2d
    urban_post_lh_urb2d=lh_urb2d
    urban_post_g_urb2d=g_urb2d
    urban_post_rn_urb2d=rn_urb2d
    urban_post_ts_urb2d=ts_urb2d
    urban_post_psim_urb2d=psim_urb2d
    urban_post_psih_urb2d=psih_urb2d
    urban_post_gz1oz0_urb2d=gz1oz0_urb2d
    urban_post_akms_urb2d=akms_urb2d
    urban_post_u10_urb2d=u10_urb2d
    urban_post_v10_urb2d=v10_urb2d
    urban_post_th2_urb2d=th2_urb2d
    urban_post_q2_urb2d=q2_urb2d
    urban_post_trl_urb3d=trl_urb3d
    urban_post_tbl_urb3d=tbl_urb3d
    urban_post_tgl_urb3d=tgl_urb3d
    urban_post_tgrl_urb3d=tgrl_urb3d
    urban_post_smr_urb3d=smr_urb3d
    urban_post_cosz_urb2d=cosz_urb2d
    urban_post_omg_urb2d=omg_urb2d
    urban_post_xlat_urb2d=xlat_urb2d
    urban_post_cmr_sfcdif=cmr_sfcdif
    urban_post_chr_sfcdif=chr_sfcdif
    urban_post_cmc_sfcdif=cmc_sfcdif
    urban_post_chc_sfcdif=chc_sfcdif
    urban_post_cmgr_sfcdif=cmgr_sfcdif
    urban_post_chgr_sfcdif=chgr_sfcdif
    urban_post_tr_urb2d_mosaic=tr_urb2d_mosaic
    urban_post_tb_urb2d_mosaic=tb_urb2d_mosaic
    urban_post_tg_urb2d_mosaic=tg_urb2d_mosaic
    urban_post_tc_urb2d_mosaic=tc_urb2d_mosaic
    urban_post_qc_urb2d_mosaic=qc_urb2d_mosaic
    urban_post_uc_urb2d_mosaic=uc_urb2d_mosaic
    urban_post_sh_urb2d_mosaic=sh_urb2d_mosaic
    urban_post_lh_urb2d_mosaic=lh_urb2d_mosaic
    urban_post_g_urb2d_mosaic=g_urb2d_mosaic
    urban_post_rn_urb2d_mosaic=rn_urb2d_mosaic
    urban_post_ts_urb2d_mosaic=ts_urb2d_mosaic
    urban_post_ts_rul_urb2d_mosaic=ts_rul_urb2d_mosaic
    urban_post_trl_urb3d_mosaic=trl_urb3d_mosaic
    urban_post_tbl_urb3d_mosaic=tbl_urb3d_mosaic
    urban_post_tgl_urb3d_mosaic=tgl_urb3d_mosaic
    urban_post_ust=ust
    urban_post_ust_urb2d=ust_urb2d
    urban_post_frc_urb2d=frc_urb2d
    call dump('_out')
    ! Verbatim surface_driver.F:3004-3016, after restoring dominant IVGTYP.
    j=1
    do i=1,ncase
#include "mosaic_surface_driver_3004_3016.inc"
    enddo
    call dump('_override')
    ! Independent tile increments: a public one-tile WRF call with copied
    ! PRE-step tile state, identical forcing, zero accumulators and unit area.
    after_qgh=qgh
    after_glw=glw
    after_swdown=swdown
    after_rainbl=rainbl
    after_sr=sr
    after_chs=chs
    after_cqs2=cqs2
    after_chs2=chs2
    after_rib=rib
    after_vegfra=vegfra
    after_shdmin=shdmin
    after_shdmax=shdmax
    after_tmn=tmn
    after_xland=xland
    after_xice=xice
    after_snoalb=snoalb
    after_embck=embck
    after_tsk=tsk
    after_hfx=hfx
    after_qfx=qfx
    after_lh=lh
    after_grdflx=grdflx
    after_qsfc=qsfc
    after_canwat=canwat
    after_snow=snow
    after_snowc=snowc
    after_snowh=snowh
    after_albedo=albedo
    after_albbck=albbck
    after_emiss=emiss
    after_znt=znt
    after_z0=z0
    after_snotime=snotime
    after_lai=lai
    after_smstav=smstav
    after_smstot=smstot
    after_sfcrunoff=sfcrunoff
    after_udrunoff=udrunoff
    after_acsnow=acsnow
    after_acsnom=acsnom
    after_snopcx=snopcx
    after_potevp=potevp
    after_noahres=noahres
    after_chklowq=chklowq
    after_smois=smois
    after_tslb=tslb
    after_sh2o=sh2o
    after_smcrel=smcrel
    after_ivgtyp=ivgtyp
    after_tsk_mosaic=tsk_mosaic
    after_qsfc_mosaic=qsfc_mosaic
    after_canwat_mosaic=canwat_mosaic
    after_snow_mosaic=snow_mosaic
    after_snowh_mosaic=snowh_mosaic
    after_snowc_mosaic=snowc_mosaic
    after_albedo_mosaic=albedo_mosaic
    after_albbck_mosaic=albbck_mosaic
    after_emiss_mosaic=emiss_mosaic
    after_embck_mosaic=embck_mosaic
    after_znt_mosaic=znt_mosaic
    after_z0_mosaic=z0_mosaic
    after_hfx_mosaic=hfx_mosaic
    after_qfx_mosaic=qfx_mosaic
    after_lh_mosaic=lh_mosaic
    after_grdflx_mosaic=grdflx_mosaic
    after_snotime_mosaic=snotime_mosaic
    after_tslb_mosaic=tslb_mosaic
    after_smois_mosaic=smois_mosaic
    after_sh2o_mosaic=sh2o_mosaic
    if(allocated(increment_sfcrunoff))deallocate(increment_sfcrunoff)
    allocate(increment_sfcrunoff(ncase,mc_original,1));increment_sfcrunoff=0.
    if(allocated(increment_udrunoff))deallocate(increment_udrunoff)
    allocate(increment_udrunoff(ncase,mc_original,1));increment_udrunoff=0.
    if(allocated(increment_potevp))deallocate(increment_potevp)
    allocate(increment_potevp(ncase,mc_original,1));increment_potevp=0.
    if(allocated(increment_acsnom))deallocate(increment_acsnom)
    allocate(increment_acsnom(ncase,mc_original,1));increment_acsnom=0.
    if(allocated(increment_snopcx))deallocate(increment_snopcx)
    allocate(increment_snopcx(ncase,mc_original,1));increment_snopcx=0.
    if(allocated(increment_acsnow))deallocate(increment_acsnow)
    allocate(increment_acsnow(ncase,mc_original,1));increment_acsnow=0.
    do increment_t=mc_original,1,-1
      qgh=before_qgh
      glw=before_glw
      swdown=before_swdown
      rainbl=before_rainbl
      sr=before_sr
      chs=before_chs
      ! A previous urban tile writes the GRID CHS floor before this tile.
      ! Independent increments must enter with the same carried coefficient.
      if(increment_t<mc_original)then
        do i=1,ncase
          if(any(before_index(i,increment_t+1:mc_original,1)==13).or. &
             any(before_index(i,increment_t+1:mc_original,1)>=51))then
            if(chs(i,1)<.01)chs(i,1)=.01
          endif
        enddo
      endif
      cqs2=before_cqs2
      chs2=before_chs2
      rib=before_rib
      vegfra=before_vegfra
      shdmin=before_shdmin
      shdmax=before_shdmax
      tmn=before_tmn
      xland=before_xland
      xice=before_xice
      snoalb=before_snoalb
      embck=before_embck
      tsk=before_tsk
      hfx=before_hfx
      qfx=before_qfx
      lh=before_lh
      grdflx=before_grdflx
      qsfc=before_qsfc
      canwat=before_canwat
      snow=before_snow
      snowc=before_snowc
      snowh=before_snowh
      albedo=before_albedo
      albbck=before_albbck
      emiss=before_emiss
      znt=before_znt
      z0=before_z0
      snotime=before_snotime
      lai=before_lai
      smstav=before_smstav
      smstot=before_smstot
      sfcrunoff=before_sfcrunoff
      udrunoff=before_udrunoff
      acsnow=before_acsnow
      acsnom=before_acsnom
      snopcx=before_snopcx
      potevp=before_potevp
      noahres=before_noahres
      chklowq=before_chklowq
      smois=before_smois
      tslb=before_tslb
      sh2o=before_sh2o
      smcrel=before_smcrel
      ivgtyp=before_ivgtyp
      tr_urb2d=urban_pre_tr_urb2d
      tb_urb2d=urban_pre_tb_urb2d
      tg_urb2d=urban_pre_tg_urb2d
      tc_urb2d=urban_pre_tc_urb2d
      qc_urb2d=urban_pre_qc_urb2d
      uc_urb2d=urban_pre_uc_urb2d
      xxxr_urb2d=urban_pre_xxxr_urb2d
      xxxb_urb2d=urban_pre_xxxb_urb2d
      xxxg_urb2d=urban_pre_xxxg_urb2d
      xxxc_urb2d=urban_pre_xxxc_urb2d
      drelr_urb2d=urban_pre_drelr_urb2d
      drelb_urb2d=urban_pre_drelb_urb2d
      drelg_urb2d=urban_pre_drelg_urb2d
      flxhumr_urb2d=urban_pre_flxhumr_urb2d
      flxhumb_urb2d=urban_pre_flxhumb_urb2d
      flxhumg_urb2d=urban_pre_flxhumg_urb2d
      cmcr_urb2d=urban_pre_cmcr_urb2d
      tgr_urb2d=urban_pre_tgr_urb2d
      sh_urb2d=urban_pre_sh_urb2d
      lh_urb2d=urban_pre_lh_urb2d
      g_urb2d=urban_pre_g_urb2d
      rn_urb2d=urban_pre_rn_urb2d
      ts_urb2d=urban_pre_ts_urb2d
      psim_urb2d=urban_pre_psim_urb2d
      psih_urb2d=urban_pre_psih_urb2d
      gz1oz0_urb2d=urban_pre_gz1oz0_urb2d
      akms_urb2d=urban_pre_akms_urb2d
      u10_urb2d=urban_pre_u10_urb2d
      v10_urb2d=urban_pre_v10_urb2d
      th2_urb2d=urban_pre_th2_urb2d
      q2_urb2d=urban_pre_q2_urb2d
      trl_urb3d=urban_pre_trl_urb3d
      tbl_urb3d=urban_pre_tbl_urb3d
      tgl_urb3d=urban_pre_tgl_urb3d
      tgrl_urb3d=urban_pre_tgrl_urb3d
      smr_urb3d=urban_pre_smr_urb3d
      cosz_urb2d=urban_pre_cosz_urb2d
      omg_urb2d=urban_pre_omg_urb2d
      xlat_urb2d=urban_pre_xlat_urb2d
      cmr_sfcdif=urban_pre_cmr_sfcdif
      chr_sfcdif=urban_pre_chr_sfcdif
      cmc_sfcdif=urban_pre_cmc_sfcdif
      chc_sfcdif=urban_pre_chc_sfcdif
      cmgr_sfcdif=urban_pre_cmgr_sfcdif
      chgr_sfcdif=urban_pre_chgr_sfcdif
      tr_urb2d_mosaic=urban_pre_tr_urb2d_mosaic(:,increment_t:increment_t,:)
      tb_urb2d_mosaic=urban_pre_tb_urb2d_mosaic(:,increment_t:increment_t,:)
      tg_urb2d_mosaic=urban_pre_tg_urb2d_mosaic(:,increment_t:increment_t,:)
      tc_urb2d_mosaic=urban_pre_tc_urb2d_mosaic(:,increment_t:increment_t,:)
      qc_urb2d_mosaic=urban_pre_qc_urb2d_mosaic(:,increment_t:increment_t,:)
      uc_urb2d_mosaic=urban_pre_uc_urb2d_mosaic(:,increment_t:increment_t,:)
      sh_urb2d_mosaic=urban_pre_sh_urb2d_mosaic(:,increment_t:increment_t,:)
      lh_urb2d_mosaic=urban_pre_lh_urb2d_mosaic(:,increment_t:increment_t,:)
      g_urb2d_mosaic=urban_pre_g_urb2d_mosaic(:,increment_t:increment_t,:)
      rn_urb2d_mosaic=urban_pre_rn_urb2d_mosaic(:,increment_t:increment_t,:)
      ts_urb2d_mosaic=urban_pre_ts_urb2d_mosaic(:,increment_t:increment_t,:)
      ts_rul_urb2d_mosaic=urban_pre_ts_rul_urb2d_mosaic(:,increment_t:increment_t,:)
      trl_urb3d_mosaic=urban_pre_trl_urb3d_mosaic(:,4*(increment_t-1)+1:4*increment_t,:)
      tbl_urb3d_mosaic=urban_pre_tbl_urb3d_mosaic(:,4*(increment_t-1)+1:4*increment_t,:)
      tgl_urb3d_mosaic=urban_pre_tgl_urb3d_mosaic(:,4*(increment_t-1)+1:4*increment_t,:)
      ust=urban_pre_ust
      ust_urb2d=urban_pre_ust_urb2d
      frc_urb2d=urban_pre_frc_urb2d
      tsk_mosaic(:,1,1)=before_tsk_mosaic(:,increment_t,1)
      qsfc_mosaic(:,1,1)=before_qsfc_mosaic(:,increment_t,1)
      canwat_mosaic(:,1,1)=before_canwat_mosaic(:,increment_t,1)
      snow_mosaic(:,1,1)=before_snow_mosaic(:,increment_t,1)
      snowh_mosaic(:,1,1)=before_snowh_mosaic(:,increment_t,1)
      snowc_mosaic(:,1,1)=before_snowc_mosaic(:,increment_t,1)
      albedo_mosaic(:,1,1)=before_albedo_mosaic(:,increment_t,1)
      albbck_mosaic(:,1,1)=before_albbck_mosaic(:,increment_t,1)
      emiss_mosaic(:,1,1)=before_emiss_mosaic(:,increment_t,1)
      embck_mosaic(:,1,1)=before_embck_mosaic(:,increment_t,1)
      znt_mosaic(:,1,1)=before_znt_mosaic(:,increment_t,1)
      z0_mosaic(:,1,1)=before_z0_mosaic(:,increment_t,1)
      hfx_mosaic(:,1,1)=before_hfx_mosaic(:,increment_t,1)
      qfx_mosaic(:,1,1)=before_qfx_mosaic(:,increment_t,1)
      lh_mosaic(:,1,1)=before_lh_mosaic(:,increment_t,1)
      grdflx_mosaic(:,1,1)=before_grdflx_mosaic(:,increment_t,1)
      snotime_mosaic(:,1,1)=before_snotime_mosaic(:,increment_t,1)
      tslb_mosaic(:,1:4,1)=before_tslb_mosaic(:,4*(increment_t-1)+1:4*increment_t,1)
      smois_mosaic(:,1:4,1)=before_smois_mosaic(:,4*(increment_t-1)+1:4*increment_t,1)
      sh2o_mosaic(:,1:4,1)=before_sh2o_mosaic(:,4*(increment_t-1)+1:4*increment_t,1)
      mosaic_cat_index=before_index
      mosaic_cat_index(:,1,1)=before_index(:,increment_t,1)
      landusef2=0.;landusef2(:,1,1)=1.
      landusef=0.
      do increment_i=1,ncase
        landusef(increment_i,mosaic_cat_index(increment_i,1,1),1)=1.
        ! Under RDLAI2D a preceding glacial tile leaves LAI=0.01.
        if(rdlai2d.and.increment_t<mc_original)then
          if(any(before_index(increment_i,increment_t+1:mc_original,1)==15))lai(increment_i,1)=.01
        endif
      enddo
      sfcrunoff=0.;udrunoff=0.;potevp=0.;acsnom=0.;snopcx=0.;acsnow=0.
      mc=1
  call lsm_mosaic(dz8w, qv3d, p8w3d, t3d, tsk,                              &
           hfx, qfx, lh, grdflx, qgh, gsw, swdown,   &
           glw, smstav, smstot,                                      &
           sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
           albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
           snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
           nsoil, dt, dzs, itimestep,                                &
           smois, tslb, snow, canwat,                                &
           chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
           .false., frpcpn,                                          &
           sh2o, snowh,                                              &
           shdavg = vegfra, snoalb = snoalb, shdmin = shdmin, shdmax = shdmax,        &
           snotime = snotime,                                        &
           acsnom = acsnom, acsnow = acsnow,                         &
           snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
           xice_threshold = 0.5,                                     &
           rdlai2d = rdlai2d, usemonalb = usemonalb,                 &
           rib = rib, noahres = noahres, opt_thcnd = opt_thcnd,      &
           ua_phys = .false., flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
           fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
           ids = 1, ide = ncase+1, jds = 1, jde = 2, kds = 1, kde = nlev, &
           ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = nlev, &
           its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = nlev-1, &
           use_wudapt_lcz=use_lcz, slucm_distributed_drag=.false., sf_urban_physics = sf_urban_physics, ust_urb2d = ust_urb2d,              &
           num_roof_layers = nsoil, num_wall_layers = nsoil,                 &
           num_road_layers = nsoil, julian = 1, julyr = 1974,            &
           frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
           num_urban_ndm = num_urban_ndm, urban_map_zrd = urban_map_zrd, urban_map_zwd = urban_map_zwd,  &
           urban_map_gd = urban_map_gd, urban_map_zd = urban_map_zd, urban_map_zdf = urban_map_zdf,    &
           urban_map_bd = urban_map_bd, urban_map_wd = urban_map_wd, urban_map_gbd = urban_map_gbd,    &
           urban_map_fbd = urban_map_fbd, urban_map_zgrd = urban_map_zgrd, num_urban_hi = num_urban_hi,  &
           tr_urb2d = tr_urb2d, &
           tb_urb2d = tb_urb2d, &
           tg_urb2d = tg_urb2d, &
           tc_urb2d = tc_urb2d, &
           qc_urb2d = qc_urb2d, &
           xxxr_urb2d = xxxr_urb2d, &
           xxxb_urb2d = xxxb_urb2d, &
           xxxg_urb2d = xxxg_urb2d, &
           xxxc_urb2d = xxxc_urb2d, &
           drelr_urb2d = drelr_urb2d, &
           drelb_urb2d = drelb_urb2d, &
           drelg_urb2d = drelg_urb2d, &
           flxhumr_urb2d = flxhumr_urb2d, &
           flxhumb_urb2d = flxhumb_urb2d, &
           flxhumg_urb2d = flxhumg_urb2d, &
           cmcr_urb2d = cmcr_urb2d, &
           tgr_urb2d = tgr_urb2d, &
           trl_urb3d = trl_urb3d, &
           tbl_urb3d = tbl_urb3d, &
           tgl_urb3d = tgl_urb3d, &
           tgrl_urb3d = tgrl_urb3d, &
           smr_urb3d = smr_urb3d, &
           sh_urb2d = sh_urb2d, &
           lh_urb2d = lh_urb2d, &
           g_urb2d = g_urb2d, &
           rn_urb2d = rn_urb2d, &
           ts_urb2d = ts_urb2d, &
           trb_urb4d = trb_urb4d, &
           tw1_urb4d = tw1_urb4d, &
           tw2_urb4d = tw2_urb4d, &
           tgb_urb4d = tgb_urb4d, &
           tlev_urb3d = tlev_urb3d, &
           qlev_urb3d = qlev_urb3d, &
           tw1lev_urb3d = tw1lev_urb3d, &
           tw2lev_urb3d = tw2lev_urb3d, &
           tglev_urb3d = tglev_urb3d, &
           tflev_urb3d = tflev_urb3d, &
           lf_ac_urb3d = lf_ac_urb3d, &
           sf_ac_urb3d = sf_ac_urb3d, &
           cm_ac_urb3d = cm_ac_urb3d, &
           sfvent_urb3d = sfvent_urb3d, &
           lfvent_urb3d = lfvent_urb3d, &
           sfwin1_urb3d = sfwin1_urb3d, &
           sfwin2_urb3d = sfwin2_urb3d, &
           sfw1_urb3d = sfw1_urb3d, &
           sfw2_urb3d = sfw2_urb3d, &
           sfr_urb3d = sfr_urb3d, &
           sfg_urb3d = sfg_urb3d, &
           ep_pv_urb3d = ep_pv_urb3d, &
           t_pv_urb3d = t_pv_urb3d, &
           trv_urb4d = trv_urb4d, &
           qr_urb4d = qr_urb4d, &
           qgr_urb3d = qgr_urb3d, &
           tgr_urb3d = tgr_urb3d, &
           drain_urb4d = drain_urb4d, &
           draingr_urb3d = draingr_urb3d, &
           sfrv_urb3d = sfrv_urb3d, &
           lfrv_urb3d = lfrv_urb3d, &
           dgr_urb3d = dgr_urb3d, &
           dg_urb3d = dg_urb3d, &
           lfr_urb3d = lfr_urb3d, &
           lfg_urb3d = lfg_urb3d, &
           hi_urb2d = hi_urb2d, &
           lp_urb2d = lp_urb2d, &
           lb_urb2d = lb_urb2d, &
           hgt_urb2d = hgt_urb2d, &
           mh_urb2d = mh_urb2d, &
           stdh_urb2d = stdh_urb2d, &
           lf_urb2d = lf_urb2d, &
           a_u_bep = a_u_bep, &
           a_v_bep = a_v_bep, &
           a_t_bep = a_t_bep, &
           a_q_bep = a_q_bep, &
           a_e_bep = a_e_bep, &
           b_u_bep = b_u_bep, &
           b_v_bep = b_v_bep, &
           b_t_bep = b_t_bep, &
           b_q_bep = b_q_bep, &
           b_e_bep = b_e_bep, &
           vl_bep = vl_bep, &
           dlg_bep = dlg_bep, &
           sf_bep = sf_bep, &
           dl_u_bep = dl_u_bep, &
           uc_urb2d = uc_urb2d, &
           psim_urb2d = psim_urb2d, &
           psih_urb2d = psih_urb2d, &
           u10_urb2d = u10_urb2d, &
           v10_urb2d = v10_urb2d, &
           gz1oz0_urb2d = gz1oz0_urb2d, &
           akms_urb2d = akms_urb2d, &
           th2_urb2d = th2_urb2d, &
           q2_urb2d = q2_urb2d, &
           cosz_urb2d = cosz_urb2d, &
           omg_urb2d = omg_urb2d, &
           xlat_urb2d = xlat_urb2d, &
           lf_urb2d_s = lf_urb2d_s, &
           z0_urb2d = z0_urb2d, &
           ust = ust, &
           tsk_rural_bep = tsk_rural_bep, &
           xlong = xlong, &
           xlat = xlat, &
           u_phy = u_phy, &
           v_phy = v_phy, &
           th_phy = th_phy, &
           rho = rho, &
           p_phy = p_phy, &
           cmr_sfcdif=cmr_sfcdif, &
           chr_sfcdif=chr_sfcdif, &
           cmc_sfcdif=cmc_sfcdif, &
           chc_sfcdif=chc_sfcdif, &
           cmgr_sfcdif=cmgr_sfcdif, &
           chgr_sfcdif=chgr_sfcdif, &
           dzr=dzr, dzb=dzb, dzg=dzg, declin_urb=0.2, gmt=12.0, julday=180, &
 nlcat=nlcat, &
 landusef=landusef, &
 landusef2=landusef2, &
 sf_surface_mosaic=1, &
 mosaic_cat=mc, &
 mosaic_cat_index=mosaic_cat_index, &
 tsk_mosaic=tsk_mosaic, &
 qsfc_mosaic=qsfc_mosaic, &
 canwat_mosaic=canwat_mosaic, &
 snow_mosaic=snow_mosaic, &
 snowh_mosaic=snowh_mosaic, &
 snowc_mosaic=snowc_mosaic, &
 albedo_mosaic=albedo_mosaic, &
 albbck_mosaic=albbck_mosaic, &
 emiss_mosaic=emiss_mosaic, &
 embck_mosaic=embck_mosaic, &
 znt_mosaic=znt_mosaic, &
 z0_mosaic=z0_mosaic, &
 hfx_mosaic=hfx_mosaic, &
 qfx_mosaic=qfx_mosaic, &
 lh_mosaic=lh_mosaic, &
 grdflx_mosaic=grdflx_mosaic, &
 snotime_mosaic=snotime_mosaic, &
 rc_mosaic=rc_mosaic, &
 lai_mosaic=lai_mosaic, &
 tr_urb2d_mosaic=tr_urb2d_mosaic, &
 tb_urb2d_mosaic=tb_urb2d_mosaic, &
 tg_urb2d_mosaic=tg_urb2d_mosaic, &
 tc_urb2d_mosaic=tc_urb2d_mosaic, &
 qc_urb2d_mosaic=qc_urb2d_mosaic, &
 uc_urb2d_mosaic=uc_urb2d_mosaic, &
 sh_urb2d_mosaic=sh_urb2d_mosaic, &
 lh_urb2d_mosaic=lh_urb2d_mosaic, &
 g_urb2d_mosaic=g_urb2d_mosaic, &
 rn_urb2d_mosaic=rn_urb2d_mosaic, &
 ts_urb2d_mosaic=ts_urb2d_mosaic, &
 ts_rul2d_mosaic=ts_rul_urb2d_mosaic, &
 tslb_mosaic=tslb_mosaic, &
 smois_mosaic=smois_mosaic, &
 sh2o_mosaic=sh2o_mosaic, &
 trl_urb3d_mosaic=trl_urb3d_mosaic, &
 tbl_urb3d_mosaic=tbl_urb3d_mosaic, &
 tgl_urb3d_mosaic=tgl_urb3d_mosaic, &
 sda_hfx=sda_hfx,sda_qfx=sda_qfx,hfx_both=hfx_both,qfx_both=qfx_both,qnorm=qnorm,fasdas=0,rc2=rc2,xlai2=xlai2)
      increment_sfcrunoff(:,increment_t,1)=sfcrunoff(:,1)
      increment_udrunoff(:,increment_t,1)=udrunoff(:,1)
      increment_potevp(:,increment_t,1)=potevp(:,1)
      increment_acsnom(:,increment_t,1)=acsnom(:,1)
      increment_snopcx(:,increment_t,1)=snopcx(:,1)
      increment_acsnow(:,increment_t,1)=acsnow(:,1)
    enddo
    mc=mc_original
    qgh=after_qgh
    glw=after_glw
    swdown=after_swdown
    rainbl=after_rainbl
    sr=after_sr
    chs=after_chs
    cqs2=after_cqs2
    chs2=after_chs2
    rib=after_rib
    vegfra=after_vegfra
    shdmin=after_shdmin
    shdmax=after_shdmax
    tmn=after_tmn
    xland=after_xland
    xice=after_xice
    snoalb=after_snoalb
    embck=after_embck
    tsk=after_tsk
    hfx=after_hfx
    qfx=after_qfx
    lh=after_lh
    grdflx=after_grdflx
    qsfc=after_qsfc
    canwat=after_canwat
    snow=after_snow
    snowc=after_snowc
    snowh=after_snowh
    albedo=after_albedo
    albbck=after_albbck
    emiss=after_emiss
    znt=after_znt
    z0=after_z0
    snotime=after_snotime
    lai=after_lai
    smstav=after_smstav
    smstot=after_smstot
    sfcrunoff=after_sfcrunoff
    udrunoff=after_udrunoff
    acsnow=after_acsnow
    acsnom=after_acsnom
    snopcx=after_snopcx
    potevp=after_potevp
    noahres=after_noahres
    chklowq=after_chklowq
    smois=after_smois
    tslb=after_tslb
    sh2o=after_sh2o
    smcrel=after_smcrel
    ivgtyp=after_ivgtyp
    tsk_mosaic=after_tsk_mosaic
    qsfc_mosaic=after_qsfc_mosaic
    canwat_mosaic=after_canwat_mosaic
    snow_mosaic=after_snow_mosaic
    snowh_mosaic=after_snowh_mosaic
    snowc_mosaic=after_snowc_mosaic
    albedo_mosaic=after_albedo_mosaic
    albbck_mosaic=after_albbck_mosaic
    emiss_mosaic=after_emiss_mosaic
    embck_mosaic=after_embck_mosaic
    znt_mosaic=after_znt_mosaic
    z0_mosaic=after_z0_mosaic
    hfx_mosaic=after_hfx_mosaic
    qfx_mosaic=after_qfx_mosaic
    lh_mosaic=after_lh_mosaic
    grdflx_mosaic=after_grdflx_mosaic
    snotime_mosaic=after_snotime_mosaic
    tslb_mosaic=after_tslb_mosaic
    smois_mosaic=after_smois_mosaic
    sh2o_mosaic=after_sh2o_mosaic
    mosaic_cat_index=before_index;landusef2=before_fractions;landusef=before_landusef
    call oracle_put('increment_sfcrunoff',increment_sfcrunoff)
    call oracle_put('increment_udrunoff',increment_udrunoff)
    call oracle_put('increment_potevp',increment_potevp)
    call oracle_put('increment_acsnom',increment_acsnom)
    call oracle_put('increment_snopcx',increment_snopcx)
    call oracle_put('increment_acsnow',increment_acsnow)


    tr_urb2d=urban_post_tr_urb2d
    tb_urb2d=urban_post_tb_urb2d
    tg_urb2d=urban_post_tg_urb2d
    tc_urb2d=urban_post_tc_urb2d
    qc_urb2d=urban_post_qc_urb2d
    uc_urb2d=urban_post_uc_urb2d
    xxxr_urb2d=urban_post_xxxr_urb2d
    xxxb_urb2d=urban_post_xxxb_urb2d
    xxxg_urb2d=urban_post_xxxg_urb2d
    xxxc_urb2d=urban_post_xxxc_urb2d
    drelr_urb2d=urban_post_drelr_urb2d
    drelb_urb2d=urban_post_drelb_urb2d
    drelg_urb2d=urban_post_drelg_urb2d
    flxhumr_urb2d=urban_post_flxhumr_urb2d
    flxhumb_urb2d=urban_post_flxhumb_urb2d
    flxhumg_urb2d=urban_post_flxhumg_urb2d
    cmcr_urb2d=urban_post_cmcr_urb2d
    tgr_urb2d=urban_post_tgr_urb2d
    sh_urb2d=urban_post_sh_urb2d
    lh_urb2d=urban_post_lh_urb2d
    g_urb2d=urban_post_g_urb2d
    rn_urb2d=urban_post_rn_urb2d
    ts_urb2d=urban_post_ts_urb2d
    psim_urb2d=urban_post_psim_urb2d
    psih_urb2d=urban_post_psih_urb2d
    gz1oz0_urb2d=urban_post_gz1oz0_urb2d
    akms_urb2d=urban_post_akms_urb2d
    u10_urb2d=urban_post_u10_urb2d
    v10_urb2d=urban_post_v10_urb2d
    th2_urb2d=urban_post_th2_urb2d
    q2_urb2d=urban_post_q2_urb2d
    trl_urb3d=urban_post_trl_urb3d
    tbl_urb3d=urban_post_tbl_urb3d
    tgl_urb3d=urban_post_tgl_urb3d
    tgrl_urb3d=urban_post_tgrl_urb3d
    smr_urb3d=urban_post_smr_urb3d
    cosz_urb2d=urban_post_cosz_urb2d
    omg_urb2d=urban_post_omg_urb2d
    xlat_urb2d=urban_post_xlat_urb2d
    cmr_sfcdif=urban_post_cmr_sfcdif
    chr_sfcdif=urban_post_chr_sfcdif
    cmc_sfcdif=urban_post_cmc_sfcdif
    chc_sfcdif=urban_post_chc_sfcdif
    cmgr_sfcdif=urban_post_cmgr_sfcdif
    chgr_sfcdif=urban_post_chgr_sfcdif
    tr_urb2d_mosaic=urban_post_tr_urb2d_mosaic
    tb_urb2d_mosaic=urban_post_tb_urb2d_mosaic
    tg_urb2d_mosaic=urban_post_tg_urb2d_mosaic
    tc_urb2d_mosaic=urban_post_tc_urb2d_mosaic
    qc_urb2d_mosaic=urban_post_qc_urb2d_mosaic
    uc_urb2d_mosaic=urban_post_uc_urb2d_mosaic
    sh_urb2d_mosaic=urban_post_sh_urb2d_mosaic
    lh_urb2d_mosaic=urban_post_lh_urb2d_mosaic
    g_urb2d_mosaic=urban_post_g_urb2d_mosaic
    rn_urb2d_mosaic=urban_post_rn_urb2d_mosaic
    ts_urb2d_mosaic=urban_post_ts_urb2d_mosaic
    ts_rul_urb2d_mosaic=urban_post_ts_rul_urb2d_mosaic
    trl_urb3d_mosaic=urban_post_trl_urb3d_mosaic
    tbl_urb3d_mosaic=urban_post_tbl_urb3d_mosaic
    tgl_urb3d_mosaic=urban_post_tgl_urb3d_mosaic
    ust=urban_post_ust
    ust_urb2d=urban_post_ust_urb2d
    frc_urb2d=urban_post_frc_urb2d
    call oracle_close()
  enddo
  deallocate(landusef,landusef2,mosaic_cat_index)
  deallocate(tsk_mosaic)
  deallocate(qsfc_mosaic)
  deallocate(canwat_mosaic)
  deallocate(snow_mosaic)
  deallocate(snowh_mosaic)
  deallocate(snowc_mosaic)
  deallocate(albedo_mosaic)
  deallocate(albbck_mosaic)
  deallocate(emiss_mosaic)
  deallocate(embck_mosaic)
  deallocate(znt_mosaic)
  deallocate(z0_mosaic)
  deallocate(hfx_mosaic)
  deallocate(qfx_mosaic)
  deallocate(lh_mosaic)
  deallocate(grdflx_mosaic)
  deallocate(snotime_mosaic)
  deallocate(rc_mosaic)
  deallocate(lai_mosaic)
  deallocate(tr_urb2d_mosaic)
  deallocate(tb_urb2d_mosaic)
  deallocate(tg_urb2d_mosaic)
  deallocate(tc_urb2d_mosaic)
  deallocate(qc_urb2d_mosaic)
  deallocate(uc_urb2d_mosaic)
  deallocate(sh_urb2d_mosaic)
  deallocate(lh_urb2d_mosaic)
  deallocate(g_urb2d_mosaic)
  deallocate(rn_urb2d_mosaic)
  deallocate(ts_urb2d_mosaic)
  deallocate(ts_rul_urb2d_mosaic)
  deallocate(tslb_mosaic)
  deallocate(smois_mosaic)
  deallocate(sh2o_mosaic)
  deallocate(trl_urb3d_mosaic)
  deallocate(tbl_urb3d_mosaic)
  deallocate(tgl_urb3d_mosaic)
  enddo
contains
  subroutine build_fixture()
    integer :: i

    do i = 1, ncase
      ivgtyp(i, 1) = 10          ! MODIS grassland
      isltyp(i, 1) = 8           ! silty clay loam
      p8w3d(i, 1, 1) = 98000.0   ! surface pressure
      p8w3d(i, 2, 1) = 97000.0   ! -> SFCPRS = 97500 exactly
      t3d(i, 1, 1) = 290.0
      t3d(i, 2, 1) = 289.0
      qv3d(i, 1, 1) = 0.008
      qv3d(i, 2, 1) = 0.008
      dz8w(i, 1, 1) = 60.0
      dz8w(i, 2, 1) = 60.0
      qgh(i, 1) = 0.012
      gsw(i, 1) = 400.0
      swdown(i, 1) = 500.0
      swddir(i, 1) = 400.0
      swddif(i, 1) = 100.0
      glw(i, 1) = 330.0
      rainbl(i, 1) = 0.0
      sr(i, 1) = 0.0
      chs(i, 1) = 0.02
      chs2(i, 1) = 0.03
      cqs2(i, 1) = 0.03
      cpm(i, 1) = 1010.0
      qz0(i, 1) = 0.005
      rib(i, 1) = -0.1
      vegfra(i, 1) = 60.0
      shdmin(i, 1) = 10.0
      shdmax(i, 1) = 80.0
      tmn(i, 1) = 285.0
      xland(i, 1) = 1.0
      xice(i, 1) = 0.0
      snoalb(i, 1) = 0.7
      embck(i, 1) = 0.95
      tsk(i, 1) = 291.0
      hfx(i, 1) = 0.0
      qfx(i, 1) = 0.0
      lh(i, 1) = 0.0
      grdflx(i, 1) = 0.0
      qsfc(i, 1) = 0.010
      canwat(i, 1) = 0.3
      snow(i, 1) = 0.0
      snowc(i, 1) = 0.0
      snowh(i, 1) = 0.0
      albedo(i, 1) = 0.2
      albbck(i, 1) = 0.2
      emiss(i, 1) = 0.95
      znt(i, 1) = 0.1
      z0(i, 1) = 0.1
      snotime(i, 1) = 0.0
      lai(i, 1) = 3.0
      smstav(i, 1) = 0.0
      smstot(i, 1) = 0.0
      sfcrunoff(i, 1) = 0.0
      udrunoff(i, 1) = 0.0
      acsnom(i, 1) = 0.0
      acsnow(i, 1) = 0.0
      snopcx(i, 1) = 0.0
      potevp(i, 1) = 0.0
      frc_urb2d(i, 1) = 0.0
      utype_urb2d(i, 1) = 1
      sda_hfx(i, 1) = 0.0
      sda_qfx(i, 1) = 0.0
      hfx_both(i, 1) = 0.0
      qfx_both(i, 1) = 0.0
      qnorm(i, 1) = 0.0
      do ns = 1, nsoil
        smois(i, ns, 1) = 0.28
        sh2o(i, ns, 1) = 0.28
      end do
      tslb(i, 1, 1) = 289.0
      tslb(i, 2, 1) = 288.0
      tslb(i, 3, 1) = 287.0
      tslb(i, 4, 1) = 286.0
    end do


    smois(2, :, 1) = 0.06
    sh2o(2, :, 1) = 0.06
    vegfra(2, 1) = 25.0

    smois(3, :, 1) = 0.46
    sh2o(3, :, 1) = 0.46
    rainbl(3, 1) = 6.0

    vegfra(4, 1) = 0.0
    shdmin(4, 1) = 0.0
    lai(4, 1) = 0.0
    canwat(4, 1) = 0.0

    vegfra(5, 1) = 100.0
    shdmin(5, 1) = 90.0
    shdmax(5, 1) = 100.0
    canwat(5, 1) = 0.5
    lai(5, 1) = 5.0
    ivgtyp(5, 1) = 1           ! evergreen needleleaf

    t3d(6, 1, 1) = 265.0
    tsk(6, 1) = 263.0
    snow(6, 1) = 20.0
    snowh(6, 1) = 0.08
    snowc(6, 1) = 0.9
    tslb(6, :, 1) = 268.0
    sh2o(6, :, 1) = 0.10
    glw(6, 1) = 250.0
    swdown(6, 1) = 200.0

    t3d(7, 1, 1) = 274.0
    tsk(7, 1) = 273.5
    snow(7, 1) = 10.0
    snowh(7, 1) = 0.04
    snowc(7, 1) = 0.7
    tslb(7, :, 1) = 272.5
    sh2o(7, :, 1) = 0.20

    t3d(8, 1, 1) = 255.0
    tsk(8, 1) = 250.0
    snow(8, 1) = 200.0
    snowh(8, 1) = 0.60
    snowc(8, 1) = 1.0
    tslb(8, :, 1) = 260.0
    sh2o(8, :, 1) = 0.05
    snotime(8, 1) = 172800.0
    swdown(8, 1) = 120.0
    glw(8, 1) = 200.0

    tslb(9, 1, 1) = 265.0
    tslb(9, 2, 1) = 268.0
    tslb(9, 3, 1) = 270.0
    tslb(9, 4, 1) = 272.0
    sh2o(9, :, 1) = 0.05
    t3d(9, 1, 1) = 270.0
    tsk(9, 1) = 268.0

    t3d(10, 1, 1) = 273.15
    rainbl(10, 1) = 1.0
    tsk(10, 1) = 272.0

    t3d(11, 1, 1) = nearest(273.15, 1.0)
    rainbl(11, 1) = 1.0
    tsk(11, 1) = 272.0

    tsk(12, 1) = 273.14
    snow(12, 1) = 5.0
    snowh(12, 1) = 0.02
    snowc(12, 1) = 0.5
    t3d(12, 1, 1) = 272.0
    tslb(12, :, 1) = 272.0

    tsk(13, 1) = 273.0
    swdown(13, 1) = 0.0
    gsw(13, 1) = 0.0
    snow(13, 1) = 5.0
    snowh(13, 1) = 0.02
    snowc(13, 1) = 0.5
    t3d(13, 1, 1) = 272.0
    tslb(13, :, 1) = 272.0

    tsk(14, 1) = nearest(273.0, 1.0)
    swdown(14, 1) = nearest(10.0, 1.0)
    snow(14, 1) = 5.0
    snowh(14, 1) = 0.02
    snowc(14, 1) = 0.5
    t3d(14, 1, 1) = 272.0
    tslb(14, :, 1) = 272.0

    rainbl(15, 1) = 0.0

    rainbl(16, 1) = nzero

    rainbl(17, 1) = subn

    snow(18, 1) = subn
    snowc(18, 1) = 0.1

    qv3d(19, 1, 1) = subn

    qv3d(20, 1, 1) = 0.0

    qv3d(21, 1, 1) = nzero

    vegfra(22, 1) = 0.0
    shdmin(22, 1) = 0.0

    vegfra(23, 1) = nzero
    shdmin(23, 1) = nzero

    canwat(24, 1) = subn

    chs(25, 1) = subn
    chs2(25, 1) = subn
    cqs2(25, 1) = subn

    xland(26, 1) = 2.0

    xice(27, 1) = 0.6
    t3d(27, 1, 1) = 260.0
    tsk(27, 1) = 258.0
    tslb(27, :, 1) = 262.0
    snow(27, 1) = 15.0
    snowh(27, 1) = 0.06
    snowc(27, 1) = 0.8

    ivgtyp(28, 1) = 15
    t3d(28, 1, 1) = 258.0
    tsk(28, 1) = 255.0
    tslb(28, :, 1) = 258.0
    snow(28, 1) = 100.0
    snowh(28, 1) = 0.4
    snowc(28, 1) = 1.0

    isltyp(29, 1) = 14

    ivgtyp(30, 1) = 13
    vegfra(30, 1) = 5.0

    isltyp(31, 1) = 1
    smois(31, :, 1) = 0.15
    sh2o(31, :, 1) = 0.15

    isltyp(32, 1) = 12
    smois(32, :, 1) = 0.40
    sh2o(32, :, 1) = 0.40

    isltyp(33, 1) = 13
    smois(33, :, 1) = 0.35
    sh2o(33, :, 1) = 0.35

    isltyp(34, 1) = 15
    smois(34, :, 1) = 0.08
    sh2o(34, :, 1) = 0.08

    swdown(35, 1) = 0.0
    gsw(35, 1) = 0.0
    swddir(35, 1) = 0.0
    swddif(35, 1) = 0.0
    glw(35, 1) = 280.0
    t3d(35, 1, 1) = 280.0
    tsk(35, 1) = 278.0
    chs(35, 1) = 0.002
    chs2(35, 1) = 0.003
    cqs2(35, 1) = 0.003
    rib(35, 1) = 0.3

    qgh(36, 1) = 0.030
    qv3d(36, 1, 1) = 0.002
    swdown(36, 1) = 900.0
    gsw(36, 1) = 750.0
    t3d(36, 1, 1) = 305.0
    tsk(36, 1) = 312.0

    sr(37, 1) = 0.5
    rainbl(37, 1) = 2.0
    t3d(37, 1, 1) = 272.5
    tsk(37, 1) = 272.0
    snow(37, 1) = 3.0
    snowh(37, 1) = 0.012
    snowc(37, 1) = 0.4

    snotime(38, 1) = 86400.0
    snow(38, 1) = 8.0
    snowh(38, 1) = 0.03
    snowc(38, 1) = 0.6
    t3d(38, 1, 1) = 268.0
    tsk(38, 1) = 266.0
    tslb(38, :, 1) = 270.0

    smois(39, :, 1) = 0.60
    sh2o(39, :, 1) = 0.60

    t3d(40, 1, 1) = 250.0
    tsk(40, 1) = 248.0
    tslb(40, :, 1) = 250.0
    sh2o(40, :, 1) = 0.02
    snow(40, 1) = 50.0
    snowh(40, 1) = 0.2
    snowc(40, 1) = 1.0
    swdown(40, 1) = 80.0
    glw(40, 1) = 180.0

    isltyp(41, 1) = 3          ! sandy loam
    smois(41, :, 1) = 0.22
    sh2o(41, :, 1) = 0.22

    isltyp(42, 1) = 4          ! silt loam
    smois(42, :, 1) = 0.30
    sh2o(42, :, 1) = 0.12
    tslb(42, :, 1) = 269.0
    t3d(42, 1, 1) = 266.0
    tsk(42, 1) = 264.0
    snow(42, 1) = 12.0
    snowh(42, 1) = 0.05
    snowc(42, 1) = 0.8
    swdown(42, 1) = 150.0
    glw(42, 1) = 230.0
  end subroutine build_fixture

  subroutine dump(suffix)
    character(len=*), intent(in) :: suffix
    call oracle_put('lp_urb2d'//trim(suffix),lp_urb2d)
    call oracle_put('lb_urb2d'//trim(suffix),lb_urb2d)
    call oracle_put('hgt_urb2d'//trim(suffix),hgt_urb2d)
    call oracle_put('mh_urb2d'//trim(suffix),mh_urb2d)
    call oracle_put('stdh_urb2d'//trim(suffix),stdh_urb2d)
    call oracle_put('lf_urb2d'//trim(suffix),lf_urb2d)
    call oracle_put('lf_urb2d_s'//trim(suffix),lf_urb2d_s)
    call oracle_put('z0_urb2d'//trim(suffix),z0_urb2d)
    call oracle_put('tr_urb2d'//trim(suffix),tr_urb2d)
    call oracle_put('tb_urb2d'//trim(suffix),tb_urb2d)
    call oracle_put('tg_urb2d'//trim(suffix),tg_urb2d)
    call oracle_put('tc_urb2d'//trim(suffix),tc_urb2d)
    call oracle_put('qc_urb2d'//trim(suffix),qc_urb2d)
    call oracle_put('uc_urb2d'//trim(suffix),uc_urb2d)
    call oracle_put('xxxr_urb2d'//trim(suffix),xxxr_urb2d)
    call oracle_put('xxxb_urb2d'//trim(suffix),xxxb_urb2d)
    call oracle_put('xxxg_urb2d'//trim(suffix),xxxg_urb2d)
    call oracle_put('xxxc_urb2d'//trim(suffix),xxxc_urb2d)
    call oracle_put('drelr_urb2d'//trim(suffix),drelr_urb2d)
    call oracle_put('drelb_urb2d'//trim(suffix),drelb_urb2d)
    call oracle_put('drelg_urb2d'//trim(suffix),drelg_urb2d)
    call oracle_put('flxhumr_urb2d'//trim(suffix),flxhumr_urb2d)
    call oracle_put('flxhumb_urb2d'//trim(suffix),flxhumb_urb2d)
    call oracle_put('flxhumg_urb2d'//trim(suffix),flxhumg_urb2d)
    call oracle_put('cmcr_urb2d'//trim(suffix),cmcr_urb2d)
    call oracle_put('tgr_urb2d'//trim(suffix),tgr_urb2d)
    call oracle_put('sh_urb2d'//trim(suffix),sh_urb2d)
    call oracle_put('lh_urb2d'//trim(suffix),lh_urb2d)
    call oracle_put('g_urb2d'//trim(suffix),g_urb2d)
    call oracle_put('rn_urb2d'//trim(suffix),rn_urb2d)
    call oracle_put('ts_urb2d'//trim(suffix),ts_urb2d)
    call oracle_put('psim_urb2d'//trim(suffix),psim_urb2d)
    call oracle_put('psih_urb2d'//trim(suffix),psih_urb2d)
    call oracle_put('gz1oz0_urb2d'//trim(suffix),gz1oz0_urb2d)
    call oracle_put('akms_urb2d'//trim(suffix),akms_urb2d)
    call oracle_put('u10_urb2d'//trim(suffix),u10_urb2d)
    call oracle_put('v10_urb2d'//trim(suffix),v10_urb2d)
    call oracle_put('th2_urb2d'//trim(suffix),th2_urb2d)
    call oracle_put('q2_urb2d'//trim(suffix),q2_urb2d)
    call oracle_put('ust_urb2d'//trim(suffix),ust_urb2d)
    call oracle_put('trl_urb3d'//trim(suffix),trl_urb3d)
    call oracle_put('tbl_urb3d'//trim(suffix),tbl_urb3d)
    call oracle_put('tgl_urb3d'//trim(suffix),tgl_urb3d)
    call oracle_put('tgrl_urb3d'//trim(suffix),tgrl_urb3d)
    call oracle_put('smr_urb3d'//trim(suffix),smr_urb3d)
    call oracle_put('ust'//trim(suffix),ust)
    call oracle_put('u_phy'//trim(suffix),u_phy)
    call oracle_put('v_phy'//trim(suffix),v_phy)
    call oracle_put('cosz_urb2d'//trim(suffix),cosz_urb2d)
    call oracle_put('omg_urb2d'//trim(suffix),omg_urb2d)
    call oracle_put('xlat_urb2d'//trim(suffix),xlat_urb2d)
    call oracle_put('u10'//trim(suffix),u10)
    call oracle_put('v10'//trim(suffix),v10)
    call oracle_put('psim'//trim(suffix),psim)
    call oracle_put('psih'//trim(suffix),psih)
    call oracle_put('gz1oz0'//trim(suffix),gz1oz0)
    call oracle_put('akhs'//trim(suffix),akhs)
    call oracle_put('akms'//trim(suffix),akms)
    call oracle_put('dz8w'//trim(suffix), dz8w)
    call oracle_put('qv3d'//trim(suffix), qv3d)
    call oracle_put('p8w3d'//trim(suffix), p8w3d)
    call oracle_put('t3d'//trim(suffix), t3d)
    call oracle_put('tsk'//trim(suffix), tsk)
    call oracle_put('hfx'//trim(suffix), hfx)
    call oracle_put('qfx'//trim(suffix), qfx)
    call oracle_put('lh'//trim(suffix), lh)
    call oracle_put('grdflx'//trim(suffix), grdflx)
    call oracle_put('qgh'//trim(suffix), qgh)
    call oracle_put('gsw'//trim(suffix), gsw)
    call oracle_put('swdown'//trim(suffix), swdown)
    call oracle_put('swddir'//trim(suffix), swddir)
    call oracle_put('swddif'//trim(suffix), swddif)
    call oracle_put('glw'//trim(suffix), glw)
    call oracle_put('smstav'//trim(suffix), smstav)
    call oracle_put('smstot'//trim(suffix), smstot)
    call oracle_put('sfcrunoff'//trim(suffix), sfcrunoff)
    call oracle_put('udrunoff'//trim(suffix), udrunoff)
    call oracle_put('vegfra'//trim(suffix), vegfra)
    call oracle_put('albedo'//trim(suffix), albedo)
    call oracle_put('albbck'//trim(suffix), albbck)
    call oracle_put('znt'//trim(suffix), znt)
    call oracle_put('z0'//trim(suffix), z0)
    call oracle_put('tmn'//trim(suffix), tmn)
    call oracle_put('xland'//trim(suffix), xland)
    call oracle_put('xice'//trim(suffix), xice)
    call oracle_put('emiss'//trim(suffix), emiss)
    call oracle_put('embck'//trim(suffix), embck)
    call oracle_put('snowc'//trim(suffix), snowc)
    call oracle_put('qsfc'//trim(suffix), qsfc)
    call oracle_put('rainbl'//trim(suffix), rainbl)
    call oracle_put('snow'//trim(suffix), snow)
    call oracle_put('canwat'//trim(suffix), canwat)
    call oracle_put('chs'//trim(suffix), chs)
    call oracle_put('chs2'//trim(suffix), chs2)
    call oracle_put('cqs2'//trim(suffix), cqs2)
    call oracle_put('cpm'//trim(suffix), cpm)
    call oracle_put('sr'//trim(suffix), sr)
    call oracle_put('chklowq'//trim(suffix), chklowq)
    call oracle_put('lai'//trim(suffix), lai)
    call oracle_put('qz0'//trim(suffix), qz0)
    call oracle_put('snowh'//trim(suffix), snowh)
    call oracle_put('snoalb'//trim(suffix), snoalb)
    call oracle_put('shdmin'//trim(suffix), shdmin)
    call oracle_put('shdmax'//trim(suffix), shdmax)
    call oracle_put('snotime'//trim(suffix), snotime)
    call oracle_put('acsnom'//trim(suffix), acsnom)
    call oracle_put('acsnow'//trim(suffix), acsnow)
    call oracle_put('snopcx'//trim(suffix), snopcx)
    call oracle_put('potevp'//trim(suffix), potevp)
    call oracle_put('rib'//trim(suffix), rib)
    call oracle_put('noahres'//trim(suffix), noahres)
    call oracle_put('flx4_2d'//trim(suffix), flx4_2d)
    call oracle_put('fvb_2d'//trim(suffix), fvb_2d)
    call oracle_put('fbur_2d'//trim(suffix), fbur_2d)
    call oracle_put('fgsn_2d'//trim(suffix), fgsn_2d)
    call oracle_put('ust_urb2d'//trim(suffix), ust_urb2d)
    call oracle_put('frc_urb2d'//trim(suffix), frc_urb2d)
    call oracle_put('sda_hfx'//trim(suffix), sda_hfx)
    call oracle_put('sda_qfx'//trim(suffix), sda_qfx)
    call oracle_put('hfx_both'//trim(suffix), hfx_both)
    call oracle_put('qfx_both'//trim(suffix), qfx_both)
    call oracle_put('qnorm'//trim(suffix), qnorm)
    call oracle_put('smois'//trim(suffix), smois)
    call oracle_put('tslb'//trim(suffix), tslb)
    call oracle_put('sh2o'//trim(suffix), sh2o)
    call oracle_put('smcrel'//trim(suffix), smcrel)
    call oracle_put('cmr_sfcdif'//trim(suffix), cmr_sfcdif)
    call oracle_put('chr_sfcdif'//trim(suffix), chr_sfcdif)
    call oracle_put('cmc_sfcdif'//trim(suffix), cmc_sfcdif)
    call oracle_put('chc_sfcdif'//trim(suffix), chc_sfcdif)
    call oracle_put('cmgr_sfcdif'//trim(suffix), cmgr_sfcdif)
    call oracle_put('chgr_sfcdif'//trim(suffix), chgr_sfcdif)
    call oracle_put('dzs'//trim(suffix), dzs)
    call oracle_put('ivgtyp'//trim(suffix), ivgtyp)
    call oracle_put('isltyp'//trim(suffix), isltyp)
    call oracle_put('utype_urb2d'//trim(suffix), utype_urb2d)
    call oracle_put('cmr_sfcdif'//trim(suffix), cmr_sfcdif)
    call oracle_put('chr_sfcdif'//trim(suffix), chr_sfcdif)
    call oracle_put('cmc_sfcdif'//trim(suffix), cmc_sfcdif)
    call oracle_put('chc_sfcdif'//trim(suffix), chc_sfcdif)
    call oracle_put('cmgr_sfcdif'//trim(suffix), cmgr_sfcdif)
    call oracle_put('chgr_sfcdif'//trim(suffix), chgr_sfcdif)
    call oracle_put('dzs'//trim(suffix), dzs)
    call oracle_put('dzr'//trim(suffix), dzr)
    call oracle_put('dzb'//trim(suffix), dzb)
    call oracle_put('dzg'//trim(suffix), dzg)
    call oracle_put('landusef'//trim(suffix),landusef)
    call oracle_put('landusef2'//trim(suffix),landusef2)
    call oracle_put('mosaic_cat_index'//trim(suffix),mosaic_cat_index)
    call oracle_put('tsk_mosaic'//trim(suffix),tsk_mosaic)
    call oracle_put('qsfc_mosaic'//trim(suffix),qsfc_mosaic)
    call oracle_put('canwat_mosaic'//trim(suffix),canwat_mosaic)
    call oracle_put('snow_mosaic'//trim(suffix),snow_mosaic)
    call oracle_put('snowh_mosaic'//trim(suffix),snowh_mosaic)
    call oracle_put('snowc_mosaic'//trim(suffix),snowc_mosaic)
    call oracle_put('albedo_mosaic'//trim(suffix),albedo_mosaic)
    call oracle_put('albbck_mosaic'//trim(suffix),albbck_mosaic)
    call oracle_put('emiss_mosaic'//trim(suffix),emiss_mosaic)
    call oracle_put('embck_mosaic'//trim(suffix),embck_mosaic)
    call oracle_put('znt_mosaic'//trim(suffix),znt_mosaic)
    call oracle_put('z0_mosaic'//trim(suffix),z0_mosaic)
    call oracle_put('hfx_mosaic'//trim(suffix),hfx_mosaic)
    call oracle_put('qfx_mosaic'//trim(suffix),qfx_mosaic)
    call oracle_put('lh_mosaic'//trim(suffix),lh_mosaic)
    call oracle_put('grdflx_mosaic'//trim(suffix),grdflx_mosaic)
    call oracle_put('snotime_mosaic'//trim(suffix),snotime_mosaic)
    call oracle_put('lai_mosaic'//trim(suffix),lai_mosaic)
    call oracle_put('tr_urb2d_mosaic'//trim(suffix),tr_urb2d_mosaic)
    call oracle_put('tb_urb2d_mosaic'//trim(suffix),tb_urb2d_mosaic)
    call oracle_put('tg_urb2d_mosaic'//trim(suffix),tg_urb2d_mosaic)
    call oracle_put('tc_urb2d_mosaic'//trim(suffix),tc_urb2d_mosaic)
    call oracle_put('qc_urb2d_mosaic'//trim(suffix),qc_urb2d_mosaic)
    call oracle_put('uc_urb2d_mosaic'//trim(suffix),uc_urb2d_mosaic)
    call oracle_put('sh_urb2d_mosaic'//trim(suffix),sh_urb2d_mosaic)
    call oracle_put('lh_urb2d_mosaic'//trim(suffix),lh_urb2d_mosaic)
    call oracle_put('g_urb2d_mosaic'//trim(suffix),g_urb2d_mosaic)
    call oracle_put('rn_urb2d_mosaic'//trim(suffix),rn_urb2d_mosaic)
    call oracle_put('ts_urb2d_mosaic'//trim(suffix),ts_urb2d_mosaic)
    call oracle_put('ts_rul_urb2d_mosaic'//trim(suffix),ts_rul_urb2d_mosaic)
    call oracle_put('tslb_mosaic'//trim(suffix),tslb_mosaic)
    call oracle_put('smois_mosaic'//trim(suffix),smois_mosaic)
    call oracle_put('sh2o_mosaic'//trim(suffix),sh2o_mosaic)
    call oracle_put('trl_urb3d_mosaic'//trim(suffix),trl_urb3d_mosaic)
    call oracle_put('tbl_urb3d_mosaic'//trim(suffix),tbl_urb3d_mosaic)
    call oracle_put('tgl_urb3d_mosaic'//trim(suffix),tgl_urb3d_mosaic)
  end subroutine
end program run_mosaic_ucm
