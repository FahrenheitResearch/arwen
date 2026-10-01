program run_noah_hook
  ! WRF v4.7.1 public entry points only. No expected value is transcribed.
  ! One process per case avoids saved BEP parameters and allocated UCM tables
  ! leaking between options. run_all invokes this executable in child mode.
  use module_sf_noahdrv, only: lsm, soil_veg_gen_parm
  use module_sf_noahlsm, only: LCZ_1,LCZ_2,LCZ_3,LCZ_4,LCZ_5,LCZ_6, &
       LCZ_7,LCZ_8,LCZ_9,LCZ_10,LCZ_11,NATURAL
  use module_sf_urban, only: urban_param_init, urban_var_init
  use module_bep_bem_helper, only: nurbm
  use oracle_io
  implicit none
  integer, parameter :: ncase=16, nsoil=4, nlev=21
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
  integer :: i, k, ns, sf_urban_physics, lcz, opt_thcnd, itimestep, status
  logical :: usemonalb, frpcpn, rdlai2d
  character(len=1024) :: outdir, exe
  character(len=40) :: case_name
  character(len=24), parameter :: cases(8) = [character(len=24) :: &
    'off_reference','slucm_frc0','slucm_lcz_frc0','slucm_frc', &
    'slucm_frc0_monalb','slucm_frc_monalb','bep_frc','bep_frc_monalb']
  call oracle_root(outdir)
  call get_command_argument(2,case_name)
  if (len_trim(case_name)==0) then
    call get_command_argument(0,exe)
    do i=1,size(cases)
      call execute_command_line('"'//trim(exe)//'" "'//trim(outdir)//'" '//trim(cases(i)), exitstat=status)
      if (status/=0) error stop 'child oracle failed'
    end do
    stop
  endif
  sf_urban_physics=1
  if (trim(case_name)=='off_reference') sf_urban_physics=0
  if (index(case_name,'bep_')==1) sf_urban_physics=2
  lcz=0
  if (trim(case_name)=='slucm_lcz_frc0') lcz=1
  ! WRF config setup, module_check_a_mundo.F:454-462. The BEP helper's
  ! class-count global must be initialized before its explicit-shape calls.
  nurbm=3
  if (lcz==1) nurbm=11
  opt_thcnd=2; itimestep=2; frpcpn=.false.; rdlai2d=.false.
  usemonalb=index(case_name,'monalb')>0
  call soil_veg_gen_parm('MODIFIED_IGBP_MODIS_NOAH','STAS')
  dzs=[0.10,0.30,0.60,1.00]
  call build_fixture()
  tr_urb2d=0.0
  tb_urb2d=0.0
  tg_urb2d=0.0
  tc_urb2d=0.0
  qc_urb2d=0.0
  xxxr_urb2d=0.0
  xxxb_urb2d=0.0
  xxxg_urb2d=0.0
  xxxc_urb2d=0.0
  drelr_urb2d=0.0
  drelb_urb2d=0.0
  drelg_urb2d=0.0
  flxhumr_urb2d=0.0
  flxhumb_urb2d=0.0
  flxhumg_urb2d=0.0
  cmcr_urb2d=0.0
  tgr_urb2d=0.0
  trl_urb3d=0.0
  tbl_urb3d=0.0
  tgl_urb3d=0.0
  tgrl_urb3d=0.0
  smr_urb3d=0.0
  sh_urb2d=0.0
  lh_urb2d=0.0
  g_urb2d=0.0
  rn_urb2d=0.0
  ts_urb2d=0.0
  trb_urb4d=0.0
  tw1_urb4d=0.0
  tw2_urb4d=0.0
  tgb_urb4d=0.0
  tlev_urb3d=0.0
  qlev_urb3d=0.0
  tw1lev_urb3d=0.0
  tw2lev_urb3d=0.0
  tglev_urb3d=0.0
  tflev_urb3d=0.0
  lf_ac_urb3d=0.0
  sf_ac_urb3d=0.0
  cm_ac_urb3d=0.0
  sfvent_urb3d=0.0
  lfvent_urb3d=0.0
  sfwin1_urb3d=0.0
  sfwin2_urb3d=0.0
  sfw1_urb3d=0.0
  sfw2_urb3d=0.0
  sfr_urb3d=0.0
  sfg_urb3d=0.0
  ep_pv_urb3d=0.0
  t_pv_urb3d=0.0
  trv_urb4d=0.0
  qr_urb4d=0.0
  qgr_urb3d=0.0
  tgr_urb3d=0.0
  drain_urb4d=0.0
  draingr_urb3d=0.0
  sfrv_urb3d=0.0
  lfrv_urb3d=0.0
  dgr_urb3d=0.0
  dg_urb3d=0.0
  lfr_urb3d=0.0
  lfg_urb3d=0.0
  hi_urb2d=0.0
  lp_urb2d=0.0
  lb_urb2d=0.0
  hgt_urb2d=0.0
  mh_urb2d=0.0
  stdh_urb2d=0.0
  lf_urb2d=0.0
  a_u_bep=0.0
  a_v_bep=0.0
  a_t_bep=0.0
  a_q_bep=0.0
  a_e_bep=0.0
  b_u_bep=0.0
  b_v_bep=0.0
  b_t_bep=0.0
  b_q_bep=0.0
  b_e_bep=0.0
  vl_bep=0.0
  dlg_bep=0.0
  sf_bep=0.0
  dl_u_bep=0.0
  uc_urb2d=0.0
  psim_urb2d=0.0
  psih_urb2d=0.0
  u10_urb2d=0.0
  v10_urb2d=0.0
  gz1oz0_urb2d=0.0
  akms_urb2d=0.0
  th2_urb2d=0.0
  q2_urb2d=0.0
  cosz_urb2d=0.0
  omg_urb2d=0.0
  xlat_urb2d=0.0
  lf_urb2d_s=0.0
  z0_urb2d=0.0
  ust=0.0
  tsk_rural_bep=0.0
  xlong=0.0
  xlat=0.0
  dzr=0.0; dzb=0.0; dzg=0.0
  u_phy=3.0; v_phy=1.0; rho=1.15; p_phy=p8w3d
  th_phy=t3d*(100000.0/p_phy)**r_d_over_cp
  cmr_sfcdif=0.01
  chr_sfcdif=0.01
  cmc_sfcdif=0.01
  chc_sfcdif=0.01
  cmgr_sfcdif=0.01
  chgr_sfcdif=0.01
  ust=0.3; uc_urb2d=1.0; xlat=40.0; xlong=-100.0
  xlat_urb2d=xlat; cosz_urb2d=0.7
  block
    ! Same cold-start entry points and argument order as physics_init:3294-3345.
    ! off_reference initializes identical state with option 1, then calls lsm
    ! with option 0. This isolates the urban arm from initialization changes.
    call urban_param_init(dzr,dzb,dzg,nsoil,max(1,sf_urban_physics),lcz,.false.)
    call urban_var_init( &
      isurban=13, &
      tsurface0_urb=tsk, &
      tlayer0_urb=tslb, &
      tdeep0_urb=tmn, &
      ivgtyp=ivgtyp, &
      ims=1, &
      ime=ncase, &
      jms=1, &
      jme=1, &
      kms=1, &
      kme=nlev, &
      num_soil_layers=nsoil, &
      lcz_1=LCZ_1, &
      lcz_2=LCZ_2, &
      lcz_3=LCZ_3, &
      lcz_4=LCZ_4, &
      lcz_5=LCZ_5, &
      lcz_6=LCZ_6, &
      lcz_7=LCZ_7, &
      lcz_8=LCZ_8, &
      lcz_9=LCZ_9, &
      lcz_10=LCZ_10, &
      lcz_11=LCZ_11, &
      restart=.false., &
      sf_urban_physics=max(1,sf_urban_physics), &
      xxxr_urb2d=xxxr_urb2d, &
      xxxb_urb2d=xxxb_urb2d, &
      xxxg_urb2d=xxxg_urb2d, &
      xxxc_urb2d=xxxc_urb2d, &
      tr_urb2d=tr_urb2d, &
      tb_urb2d=tb_urb2d, &
      tg_urb2d=tg_urb2d, &
      tc_urb2d=tc_urb2d, &
      qc_urb2d=qc_urb2d, &
      trl_urb3d=trl_urb3d, &
      tbl_urb3d=tbl_urb3d, &
      tgl_urb3d=tgl_urb3d, &
      sh_urb2d=sh_urb2d, &
      lh_urb2d=lh_urb2d, &
      g_urb2d=g_urb2d, &
      rn_urb2d=rn_urb2d, &
      ts_urb2d=ts_urb2d, &
      num_urban_ndm=num_urban_ndm, &
      urban_map_zrd=urban_map_zrd, &
      urban_map_zwd=urban_map_zwd, &
      urban_map_gd=urban_map_gd, &
      urban_map_zd=urban_map_zd, &
      urban_map_zdf=urban_map_zdf, &
      urban_map_bd=urban_map_bd, &
      urban_map_wd=urban_map_wd, &
      urban_map_gbd=urban_map_gbd, &
      urban_map_fbd=urban_map_fbd, &
      urban_map_zgrd=urban_map_zgrd, &
      num_urban_hi=num_urban_hi, &
      trb_urb4d=trb_urb4d, &
      tw1_urb4d=tw1_urb4d, &
      tw2_urb4d=tw2_urb4d, &
      tgb_urb4d=tgb_urb4d, &
      tlev_urb3d=tlev_urb3d, &
      qlev_urb3d=qlev_urb3d, &
      tw1lev_urb3d=tw1lev_urb3d, &
      tw2lev_urb3d=tw2lev_urb3d, &
      tglev_urb3d=tglev_urb3d, &
      tflev_urb3d=tflev_urb3d, &
      sf_ac_urb3d=sf_ac_urb3d, &
      lf_ac_urb3d=lf_ac_urb3d, &
      cm_ac_urb3d=cm_ac_urb3d, &
      sfvent_urb3d=sfvent_urb3d, &
      lfvent_urb3d=lfvent_urb3d, &
      sfwin1_urb3d=sfwin1_urb3d, &
      sfwin2_urb3d=sfwin2_urb3d, &
      sfw1_urb3d=sfw1_urb3d, &
      sfw2_urb3d=sfw2_urb3d, &
      sfr_urb3d=sfr_urb3d, &
      sfg_urb3d=sfg_urb3d, &
      ep_pv_urb3d=ep_pv_urb3d, &
      t_pv_urb3d=t_pv_urb3d, &
      trv_urb4d=trv_urb4d, &
      qr_urb4d=qr_urb4d, &
      qgr_urb3d=qgr_urb3d, &
      tgr_urb3d=tgr_urb3d, &
      drain_urb4d=drain_urb4d, &
      draingr_urb3d=draingr_urb3d, &
      sfrv_urb3d=sfrv_urb3d, &
      lfrv_urb3d=lfrv_urb3d, &
      dgr_urb3d=dgr_urb3d, &
      dg_urb3d=dg_urb3d, &
      lfr_urb3d=lfr_urb3d, &
      lfg_urb3d=lfg_urb3d, &
      smois_urb=smois, &
      lp_urb2d=lp_urb2d, &
      hi_urb2d=hi_urb2d, &
      lb_urb2d=lb_urb2d, &
      hgt_urb2d=hgt_urb2d, &
      mh_urb2d=mh_urb2d, &
      stdh_urb2d=stdh_urb2d, &
      lf_urb2d=lf_urb2d, &
      cmcr_urb2d=cmcr_urb2d, &
      tgr_urb2d=tgr_urb2d, &
      tgrl_urb3d=tgrl_urb3d, &
      smr_urb3d=smr_urb3d, &
      drelr_urb2d=drelr_urb2d, &
      drelb_urb2d=drelb_urb2d, &
      drelg_urb2d=drelg_urb2d, &
      flxhumr_urb2d=flxhumr_urb2d, &
      flxhumb_urb2d=flxhumb_urb2d, &
      flxhumg_urb2d=flxhumg_urb2d, &
      a_u_bep=a_u_bep, &
      a_v_bep=a_v_bep, &
      a_t_bep=a_t_bep, &
      a_q_bep=a_q_bep, &
      a_e_bep=a_e_bep, &
      b_u_bep=b_u_bep, &
      b_v_bep=b_v_bep, &
      b_t_bep=b_t_bep, &
      b_q_bep=b_q_bep, &
      b_e_bep=b_e_bep, &
      dlg_bep=dlg_bep, &
      dl_u_bep=dl_u_bep, &
      sf_bep=sf_bep, &
      vl_bep=vl_bep, &
      frc_urb2d=frc_urb2d, &
      utype_urb2d=utype_urb2d, &
      use_wudapt_lcz=lcz)
  end block
  ! Fraction is supplied directly to lsm after table initialization.
  frc_urb2d=0.0
  if (index(case_name,'frc0')==0.and.sf_urban_physics>0) then
    do i=1,ncase
      if(ivgtyp(i,1)/=10) frc_urb2d(i,1)=0.25+0.1*mod(i,5)
    enddo
    frc_urb2d(11,1)=0.995
    ts_urb2d=tsk+2.0
  endif
  tsk_rural_bep=tsk-1.5
  call oracle_open(trim(case_name))
  call oracle_put('option',sf_urban_physics)
  call oracle_put('use_wudapt_lcz',lcz)
  call oracle_put('usemonalb',merge(1,0,usemonalb))
  call oracle_put('opt_thcnd',opt_thcnd)
  call oracle_put('natural',NATURAL)
  call oracle_put('lcz_categories',[LCZ_1,LCZ_2,LCZ_3,LCZ_4,LCZ_5,LCZ_6,LCZ_7,LCZ_8,LCZ_9,LCZ_10,LCZ_11])
  call oracle_put('dt',dt)
  call oracle_put('isurban',13)
  call oracle_put('isice',15)
  call oracle_put('itimestep',itimestep)
  call oracle_put('frpcpn',merge(1,0,frpcpn))
  call oracle_put('rdlai2d',merge(1,0,rdlai2d))
  call oracle_put('myj',0)
  call oracle_put('ua_phys',0)
  call oracle_put('fasdas',0)
  call oracle_put('xice_threshold',0.5)
  call oracle_put('rovcp',r_d_over_cp)
  call oracle_put('julian',1)
  call oracle_put('julyr',1974)
  call oracle_put('julday',180)
  call oracle_put('declin_urb',0.2)
  call oracle_put('gmt',12.0)
  call oracle_put('nurbm',nurbm)
  call oracle_put('num_soil_layers',nsoil)
  call oracle_put('num_roof_layers',nsoil)
  call oracle_put('num_wall_layers',nsoil)
  call oracle_put('num_road_layers',nsoil)
  call oracle_put('domain_bounds',[1,ncase,1,1,1,nlev])
  call oracle_put('memory_bounds',[1,ncase,1,1,1,nlev])
  call oracle_put('tile_bounds',[1,ncase,1,1,1,nlev-1])
  call oracle_put('num_urban_ndm',num_urban_ndm)
  call oracle_put('urban_map_zrd',urban_map_zrd)
  call oracle_put('urban_map_zwd',urban_map_zwd)
  call oracle_put('urban_map_gd',urban_map_gd)
  call oracle_put('urban_map_zd',urban_map_zd)
  call oracle_put('urban_map_zdf',urban_map_zdf)
  call oracle_put('urban_map_bd',urban_map_bd)
  call oracle_put('urban_map_wd',urban_map_wd)
  call oracle_put('urban_map_gbd',urban_map_gbd)
  call oracle_put('urban_map_fbd',urban_map_fbd)
  call oracle_put('urban_map_zgrd',urban_map_zgrd)
  call oracle_put('num_urban_hi',num_urban_hi)
  call dump('_in')
  call lsm(dz8w, qv3d, p8w3d, t3d, tsk,                              &
           hfx, qfx, lh, grdflx, qgh, gsw, swdown, swddir, swddif,   &
           glw, smstav, smstot,                                      &
           sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
           albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
           snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
           nsoil, dt, dzs, itimestep,                                &
           smois, tslb, snow, canwat,                                &
           chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
           .false., frpcpn,                                          &
           sh2o, snowh,                                              &
           snoalb = snoalb, shdmin = shdmin, shdmax = shdmax,        &
           snotime = snotime,                                        &
           acsnom = acsnom, acsnow = acsnow,                         &
           snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
           xice_threshold = 0.5,                                     &
           rdlai2d = rdlai2d, usemonalb = usemonalb,                 &
           rib = rib, noahres = noahres, opt_thcnd = opt_thcnd,      &
           ua_phys = .false., flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
           fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
           ids = 1, ide = ncase, jds = 1, jde = 1, kds = 1, kde = nlev, &
           ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = nlev, &
           its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = nlev-1, &
           sf_urban_physics = sf_urban_physics, ust_urb2d = ust_urb2d,              &
           num_roof_layers = nsoil, num_wall_layers = nsoil,                 &
           num_road_layers = nsoil, julian = 1, julyr = 1974,            &
           frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
           num_urban_ndm = num_urban_ndm, urban_map_zrd = urban_map_zrd, urban_map_zwd = urban_map_zwd,  &
           urban_map_gd = urban_map_gd, urban_map_zd = urban_map_zd, urban_map_zdf = urban_map_zdf,    &
           urban_map_bd = urban_map_bd, urban_map_wd = urban_map_wd, urban_map_gbd = urban_map_gbd,    &
           urban_map_fbd = urban_map_fbd, urban_map_zgrd = urban_map_zgrd, num_urban_hi = num_urban_hi,  &
           sda_hfx = sda_hfx, sda_qfx = sda_qfx,                     &
           hfx_both = hfx_both, qfx_both = qfx_both, qnorm = qnorm,  &
           fasdas = 0, &
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
           dzr=dzr, dzb=dzb, dzg=dzg, declin_urb=0.2, gmt=12.0, julday=180)
  call dump('_out')
  call oracle_close()
contains
  subroutine build_fixture()

    do i = 1, ncase
      ivgtyp(i, 1) = 13          ! MODIS grassland
      isltyp(i, 1) = 8           ! silty clay loam
      p8w3d(i, 1, 1) = 98000.0   ! surface pressure
      p8w3d(i, 2, 1) = 97000.0   ! -> SFCPRS = 97500 exactly
      t3d(i, 1, 1) = 290.0
      t3d(i, 2, 1) = 289.0
      qv3d(i, 1, 1) = 0.008
      qv3d(i, 2, 1) = 0.008
      dz8w(i, 1, 1) = 300.0
      dz8w(i, 2, 1) = 300.0
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


    ! Repeat physical probes in both urban and grassland columns.
    do i=1,ncase
      do k=3,nlev
        p8w3d(i,k,1)=97000.0-500.0*(k-2)
        t3d(i,k,1)=289.0-0.5*(k-2)
        qv3d(i,k,1)=0.008
        dz8w(i,k,1)=300.0
      enddo
      if (mod(i,4)==0) ivgtyp(i,1)=10
      if (mod(i,3)==0) then
        swdown(i,1)=0.0; gsw(i,1)=0.0
        swddir(i,1)=0.0; swddif(i,1)=0.0
        tsk(i,1)=288.0; rib(i,1)=0.2
      endif
      if (mod(i,2)==0) rainbl(i,1)=2.0
      if (mod(i,5)==0) then
        snow(i,1)=20.0; snowh(i,1)=0.08; snowc(i,1)=0.9
        tsk(i,1)=263.0; t3d(i,:,1)=265.0
        tslb(i,:,1)=268.0; sh2o(i,:,1)=0.10
        glw(i,1)=250.0; qv3d(i,:,1)=0.002
      endif
      isltyp(i,1)=8
      if (mod(i,3)==1) isltyp(i,1)=3
      if (mod(i,3)==2) isltyp(i,1)=4
    enddo
    if (lcz==1) then
      ivgtyp(1:11,1)=[LCZ_1,LCZ_2,LCZ_3,LCZ_4,LCZ_5,LCZ_6,LCZ_7,LCZ_8,LCZ_9,LCZ_10,LCZ_11]
      ivgtyp(12,1)=13
    endif
    smcrel=0.0; noahres=0.0; chklowq=0.0
    flx4_2d=0.0; fvb_2d=0.0; fbur_2d=0.0; fgsn_2d=0.0; ust_urb2d=0.0
  end subroutine
  subroutine dump(suffix)
    character(len=*), intent(in) :: suffix
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
    call oracle_put('tr_urb2d'//trim(suffix), tr_urb2d)
    call oracle_put('tb_urb2d'//trim(suffix), tb_urb2d)
    call oracle_put('tg_urb2d'//trim(suffix), tg_urb2d)
    call oracle_put('tc_urb2d'//trim(suffix), tc_urb2d)
    call oracle_put('qc_urb2d'//trim(suffix), qc_urb2d)
    call oracle_put('xxxr_urb2d'//trim(suffix), xxxr_urb2d)
    call oracle_put('xxxb_urb2d'//trim(suffix), xxxb_urb2d)
    call oracle_put('xxxg_urb2d'//trim(suffix), xxxg_urb2d)
    call oracle_put('xxxc_urb2d'//trim(suffix), xxxc_urb2d)
    call oracle_put('drelr_urb2d'//trim(suffix), drelr_urb2d)
    call oracle_put('drelb_urb2d'//trim(suffix), drelb_urb2d)
    call oracle_put('drelg_urb2d'//trim(suffix), drelg_urb2d)
    call oracle_put('flxhumr_urb2d'//trim(suffix), flxhumr_urb2d)
    call oracle_put('flxhumb_urb2d'//trim(suffix), flxhumb_urb2d)
    call oracle_put('flxhumg_urb2d'//trim(suffix), flxhumg_urb2d)
    call oracle_put('cmcr_urb2d'//trim(suffix), cmcr_urb2d)
    call oracle_put('tgr_urb2d'//trim(suffix), tgr_urb2d)
    call oracle_put('trl_urb3d'//trim(suffix), trl_urb3d)
    call oracle_put('tbl_urb3d'//trim(suffix), tbl_urb3d)
    call oracle_put('tgl_urb3d'//trim(suffix), tgl_urb3d)
    call oracle_put('tgrl_urb3d'//trim(suffix), tgrl_urb3d)
    call oracle_put('smr_urb3d'//trim(suffix), smr_urb3d)
    call oracle_put('sh_urb2d'//trim(suffix), sh_urb2d)
    call oracle_put('lh_urb2d'//trim(suffix), lh_urb2d)
    call oracle_put('g_urb2d'//trim(suffix), g_urb2d)
    call oracle_put('rn_urb2d'//trim(suffix), rn_urb2d)
    call oracle_put('ts_urb2d'//trim(suffix), ts_urb2d)
    call oracle_put('trb_urb4d'//trim(suffix), trb_urb4d)
    call oracle_put('tw1_urb4d'//trim(suffix), tw1_urb4d)
    call oracle_put('tw2_urb4d'//trim(suffix), tw2_urb4d)
    call oracle_put('tgb_urb4d'//trim(suffix), tgb_urb4d)
    call oracle_put('tlev_urb3d'//trim(suffix), tlev_urb3d)
    call oracle_put('qlev_urb3d'//trim(suffix), qlev_urb3d)
    call oracle_put('tw1lev_urb3d'//trim(suffix), tw1lev_urb3d)
    call oracle_put('tw2lev_urb3d'//trim(suffix), tw2lev_urb3d)
    call oracle_put('tglev_urb3d'//trim(suffix), tglev_urb3d)
    call oracle_put('tflev_urb3d'//trim(suffix), tflev_urb3d)
    call oracle_put('lf_ac_urb3d'//trim(suffix), lf_ac_urb3d)
    call oracle_put('sf_ac_urb3d'//trim(suffix), sf_ac_urb3d)
    call oracle_put('cm_ac_urb3d'//trim(suffix), cm_ac_urb3d)
    call oracle_put('sfvent_urb3d'//trim(suffix), sfvent_urb3d)
    call oracle_put('lfvent_urb3d'//trim(suffix), lfvent_urb3d)
    call oracle_put('sfwin1_urb3d'//trim(suffix), sfwin1_urb3d)
    call oracle_put('sfwin2_urb3d'//trim(suffix), sfwin2_urb3d)
    call oracle_put('sfw1_urb3d'//trim(suffix), sfw1_urb3d)
    call oracle_put('sfw2_urb3d'//trim(suffix), sfw2_urb3d)
    call oracle_put('sfr_urb3d'//trim(suffix), sfr_urb3d)
    call oracle_put('sfg_urb3d'//trim(suffix), sfg_urb3d)
    call oracle_put('ep_pv_urb3d'//trim(suffix), ep_pv_urb3d)
    call oracle_put('t_pv_urb3d'//trim(suffix), t_pv_urb3d)
    call oracle_put('trv_urb4d'//trim(suffix), trv_urb4d)
    call oracle_put('qr_urb4d'//trim(suffix), qr_urb4d)
    call oracle_put('qgr_urb3d'//trim(suffix), qgr_urb3d)
    call oracle_put('tgr_urb3d'//trim(suffix), tgr_urb3d)
    call oracle_put('drain_urb4d'//trim(suffix), drain_urb4d)
    call oracle_put('draingr_urb3d'//trim(suffix), draingr_urb3d)
    call oracle_put('sfrv_urb3d'//trim(suffix), sfrv_urb3d)
    call oracle_put('lfrv_urb3d'//trim(suffix), lfrv_urb3d)
    call oracle_put('dgr_urb3d'//trim(suffix), dgr_urb3d)
    call oracle_put('dg_urb3d'//trim(suffix), dg_urb3d)
    call oracle_put('lfr_urb3d'//trim(suffix), lfr_urb3d)
    call oracle_put('lfg_urb3d'//trim(suffix), lfg_urb3d)
    call oracle_put('hi_urb2d'//trim(suffix), hi_urb2d)
    call oracle_put('lp_urb2d'//trim(suffix), lp_urb2d)
    call oracle_put('lb_urb2d'//trim(suffix), lb_urb2d)
    call oracle_put('hgt_urb2d'//trim(suffix), hgt_urb2d)
    call oracle_put('mh_urb2d'//trim(suffix), mh_urb2d)
    call oracle_put('stdh_urb2d'//trim(suffix), stdh_urb2d)
    call oracle_put('lf_urb2d'//trim(suffix), lf_urb2d)
    call oracle_put('a_u_bep'//trim(suffix), a_u_bep)
    call oracle_put('a_v_bep'//trim(suffix), a_v_bep)
    call oracle_put('a_t_bep'//trim(suffix), a_t_bep)
    call oracle_put('a_q_bep'//trim(suffix), a_q_bep)
    call oracle_put('a_e_bep'//trim(suffix), a_e_bep)
    call oracle_put('b_u_bep'//trim(suffix), b_u_bep)
    call oracle_put('b_v_bep'//trim(suffix), b_v_bep)
    call oracle_put('b_t_bep'//trim(suffix), b_t_bep)
    call oracle_put('b_q_bep'//trim(suffix), b_q_bep)
    call oracle_put('b_e_bep'//trim(suffix), b_e_bep)
    call oracle_put('vl_bep'//trim(suffix), vl_bep)
    call oracle_put('dlg_bep'//trim(suffix), dlg_bep)
    call oracle_put('sf_bep'//trim(suffix), sf_bep)
    call oracle_put('dl_u_bep'//trim(suffix), dl_u_bep)
    call oracle_put('uc_urb2d'//trim(suffix), uc_urb2d)
    call oracle_put('psim_urb2d'//trim(suffix), psim_urb2d)
    call oracle_put('psih_urb2d'//trim(suffix), psih_urb2d)
    call oracle_put('u10_urb2d'//trim(suffix), u10_urb2d)
    call oracle_put('v10_urb2d'//trim(suffix), v10_urb2d)
    call oracle_put('gz1oz0_urb2d'//trim(suffix), gz1oz0_urb2d)
    call oracle_put('akms_urb2d'//trim(suffix), akms_urb2d)
    call oracle_put('th2_urb2d'//trim(suffix), th2_urb2d)
    call oracle_put('q2_urb2d'//trim(suffix), q2_urb2d)
    call oracle_put('cosz_urb2d'//trim(suffix), cosz_urb2d)
    call oracle_put('omg_urb2d'//trim(suffix), omg_urb2d)
    call oracle_put('xlat_urb2d'//trim(suffix), xlat_urb2d)
    call oracle_put('lf_urb2d_s'//trim(suffix), lf_urb2d_s)
    call oracle_put('z0_urb2d'//trim(suffix), z0_urb2d)
    call oracle_put('ust'//trim(suffix), ust)
    call oracle_put('tsk_rural_bep'//trim(suffix), tsk_rural_bep)
    call oracle_put('xlong'//trim(suffix), xlong)
    call oracle_put('xlat'//trim(suffix), xlat)
    call oracle_put('u_phy'//trim(suffix), u_phy)
    call oracle_put('v_phy'//trim(suffix), v_phy)
    call oracle_put('th_phy'//trim(suffix), th_phy)
    call oracle_put('rho'//trim(suffix), rho)
    call oracle_put('p_phy'//trim(suffix), p_phy)
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
  end subroutine
end program
