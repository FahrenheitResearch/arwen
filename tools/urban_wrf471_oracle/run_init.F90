program run_urban_init_oracle
  ! WRF v4.7.1's urban_param_init + urban_var_init (phys/module_sf_urban.F
  ! :2043-2551 and :2557-3052), called exactly as module_physics_init.F
  ! :3294-3345 calls them, for every (sf_urban_physics, use_wudapt_lcz)
  ! pair, over one row of independent columns.
  !
  ! Every value written is a word WRF produced (the table module state after
  ! urban_param_init, the Registry arrays after urban_var_init) or a word this
  ! program put into an input before the call.  Nothing is computed here.
  !
  ! Dimensions: the urban map sizes are check_a_mundo.F:3274-3302 applied to
  ! the BEP / BEP_BEM module parameters (check_a_mundo.F:468-485); option 1
  ! keeps the Registry's derived defaults of 1, as WRF does.
  use oracle_io
  use module_sf_urban
  use module_sf_bep, only: bep_ndm_ => ndm, bep_nz_ => nz_um, &
                           bep_ng_ => ng_u, bep_nwr_ => nwr_u
  use module_sf_bep_bem, only: bem_ndm_ => ndm, bem_nz_ => nz_um, &
                               bem_ng_ => ng_u, bem_nwr_ => nwr_u, &
                               bem_nf_ => nf_u, bem_ngb_ => ngb_u, &
                               bem_nbui_ => nbui_max, bem_ngr_ => ngr_u
  implicit none

  integer, parameter :: ncol = 10, nsoil = 4, kms = 1, kme = 5
  integer, parameter :: num_urban_hi = 15
  ! MODIFIED_IGBP_MODIS_NOAH: ISURBAN 13, VEGPARM LCZ_1..LCZ_11 = 51..61.
  integer, parameter :: isurban = 13
  integer, parameter :: lcz(11) = (/ 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61 /)
  character(len=512) :: root, self, arg
  character(len=1200) :: cmd
  integer :: opt, wudapt, stat

  ! urban_param_init allocates its module tables ONCE, at the first call's
  ! ICATE (module_sf_urban.F:2134, IF .not. ALLOCATED(ZR_TBL)), exactly
  ! because WRF calls it with one table per run.  A second call with the
  ! other table would write 11 rows into 3-row arrays, so every
  ! (sf_urban_physics, use_wudapt_lcz) pair runs in its own process: with
  ! no case argument this program re-runs itself once per pair.
  call oracle_root(root)
  call get_command_argument(2, arg)
  if (len_trim(arg) == 0) then
    call get_command_argument(0, self)
    do opt = 1, 3
      do wudapt = 0, 1
        write(cmd, '(A,1X,A,1X,I0,1X,I0)') trim(self), trim(root), opt, wudapt
        call execute_command_line(trim(cmd), exitstat=stat)
        if (stat /= 0) error stop 'run_init: a case process failed'
      end do
    end do
  else
    read(arg, *) opt
    call get_command_argument(3, arg)
    read(arg, *) wudapt
    call one_case(opt, wudapt)
    call oracle_close()
  end if

contains

  subroutine one_case(sf_urban_physics, use_wudapt_lcz)
    integer, intent(in) :: sf_urban_physics, use_wudapt_lcz
    character(len=32) :: case
    real, dimension(nsoil) :: dzr, dzb, dzg
    integer :: ndm, nz, ng, nwr, nf, ngb, nbui, ngr
    integer :: zrd, zwd, gd, zd, zdf, bd, wd, gbd, fbd, zgrd
    integer :: i

    ! -- inputs ------------------------------------------------------------
    integer :: ivgtyp(ncol, 1)
    real :: tsk(ncol, 1), tmn(ncol, 1), frc(ncol, 1)
    real :: tslb(ncol, nsoil, 1), smois(ncol, nsoil, 1)
    ! -- state -------------------------------------------------------------
    real, dimension(ncol, 1) :: xxxr, xxxb, xxxg, xxxc, tr, tb, tg, tc, qc
    real, dimension(ncol, 1) :: sh, lh, g, rn, ts, lf_ac, sf_ac, cm_ac
    real, dimension(ncol, 1) :: sfvent, lfvent, ep_pv, qgr3, tgr3, draingr
    real, dimension(ncol, 1) :: lp, lb, hgt, mh, stdh, cmcr, tgr2
    real, dimension(ncol, 1) :: drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
    real, dimension(ncol, nsoil, 1) :: trl, tbl, tgl, tgrl, smr
    real, dimension(ncol, 4, 1) :: lf
    real, dimension(ncol, num_urban_hi, 1) :: hi
    real, dimension(ncol, kms:kme, 1) :: a_u, a_v, a_t, a_q, a_e, b_u, b_v
    real, dimension(ncol, kms:kme, 1) :: b_t, b_q, b_e, vl, dlg, sf, dl_u
    integer :: utype(ncol, 1)
    real, allocatable, dimension(:, :, :) :: trb, tw1, tw2, tgb, tlev, qlev
    real, allocatable, dimension(:, :, :) :: tw1lev, tw2lev, tglev, tflev
    real, allocatable, dimension(:, :, :) :: sfwin1, sfwin2, sfw1, sfw2, sfr
    real, allocatable, dimension(:, :, :) :: sfg, t_pv, trv, qr4, drain
    real, allocatable, dimension(:, :, :) :: sfrv, lfrv, dgr, dg, lfr, lfg

    write(case, '(A,I0,A,I0)') 'opt', sf_urban_physics, '_lcz', use_wudapt_lcz
    call oracle_open(trim(case))

    ! -- dimensions (check_a_mundo) -----------------------------------------
    ndm = 1; nz = 1; ng = 1; nwr = 1; nf = 1; ngb = 1; nbui = 1; ngr = 1
    if (sf_urban_physics == 2) then
      ndm = bep_ndm_; nz = bep_nz_; ng = bep_ng_; nwr = bep_nwr_
    else if (sf_urban_physics == 3) then
      ndm = bem_ndm_; nz = bem_nz_; ng = bem_ng_; nwr = bem_nwr_
      nf = bem_nf_; ngb = bem_ngb_; nbui = bem_nbui_; ngr = bem_ngr_
    end if
    zrd = ndm * nwr * nz
    zwd = ndm * nwr * nz * nbui
    gd = ndm * ng
    zd = ndm * nz * nbui
    zdf = ndm * nz
    bd = nz * nbui
    wd = ndm * nz * nbui
    gbd = ndm * ngb * nbui
    fbd = ndm * (nz - 1) * nf * nbui
    zgrd = ndm * ngr * nz
    allocate(trb(ncol, zrd, 1), tw1(ncol, zwd, 1), tw2(ncol, zwd, 1))
    allocate(tgb(ncol, gd, 1), tlev(ncol, bd, 1), qlev(ncol, bd, 1))
    allocate(tw1lev(ncol, wd, 1), tw2lev(ncol, wd, 1), tglev(ncol, gbd, 1))
    allocate(tflev(ncol, fbd, 1), sfwin1(ncol, wd, 1), sfwin2(ncol, wd, 1))
    allocate(sfw1(ncol, zd, 1), sfw2(ncol, zd, 1), sfr(ncol, zdf, 1))
    allocate(sfg(ncol, ndm, 1), t_pv(ncol, zdf, 1), trv(ncol, zgrd, 1))
    allocate(qr4(ncol, zgrd, 1), drain(ncol, zdf, 1), sfrv(ncol, zdf, 1))
    allocate(lfrv(ncol, zdf, 1), dgr(ncol, zdf, 1), dg(ncol, ndm, 1))
    allocate(lfr(ncol, zdf, 1), lfg(ncol, ndm, 1))

    ! -- tables -------------------------------------------------------------
    dzr = 0.0; dzb = 0.0; dzg = 0.0
    call urban_param_init(dzr, dzb, dzg, nsoil, sf_urban_physics, &
                          use_wudapt_lcz, .false.)
    call put_tables(dzr, dzb, dzg)

    ! -- inputs: a row of independent columns -------------------------------
    ! Urban categories of both legends, NATURAL (14), grassland (10), water
    ! (17); input FRC_URB2D covering the kept range (0, 1], zero, negative,
    ! above one, and a non-urban column carrying a fraction.
    if (use_wudapt_lcz == 0) then
      ivgtyp(:, 1) = (/ isurban, isurban, isurban, isurban, lcz(1), lcz(2), &
                        lcz(3), 10, 14, 17 /)
    else
      ivgtyp(:, 1) = (/ isurban, lcz(1), lcz(4), lcz(5), lcz(7), lcz(10), &
                        lcz(11), 10, 14, 17 /)
    end if
    frc(:, 1) = (/ 0.0, 0.37, 1.0, 1.5, -0.2, 0.0, 0.93, 0.4, 0.0, 0.0 /)
    do i = 1, ncol
      tsk(i, 1) = 283.17 + 1.37 * real(i)
      tmn(i, 1) = 281.3 + 0.11 * real(i)
      tslb(i, 1, 1) = 284.63 + 0.73 * real(i)
      tslb(i, 2, 1) = 283.29 + 0.41 * real(i)
      tslb(i, 3, 1) = 282.07 + 0.19 * real(i)
      tslb(i, 4, 1) = 281.51 + 0.07 * real(i)
      smois(i, 1, 1) = 0.213 + 0.017 * real(i)
      smois(i, 2, 1) = 0.227 + 0.013 * real(i)
      smois(i, 3, 1) = 0.241 + 0.011 * real(i)
      smois(i, 4, 1) = 0.263 + 0.007 * real(i)
    end do
    call oracle_put('ivgtyp', ivgtyp)
    call oracle_put('tsk_in', tsk)
    call oracle_put('tmn_in', tmn)
    call oracle_put('tslb_in', tslb)
    call oracle_put('smois_in', smois)
    call oracle_put('frc_in', frc)
    call oracle_put('dims', (/ ndm, nz, ng, nwr, nf, ngb, nbui, ngr, zrd, &
                               zwd, gd, zd, zdf, bd, wd, gbd, fbd, zgrd /))

    ! -- a sentinel in every inout so an array the routine leaves alone is
    !    visibly left alone (and the port must leave it alone too).
    xxxr = -7.0; xxxb = -7.0; xxxg = -7.0; xxxc = -7.0
    tr = -7.0; tb = -7.0; tg = -7.0; tc = -7.0; qc = -7.0
    sh = -7.0; lh = -7.0; g = -7.0; rn = -7.0; ts = -7.0
    lf_ac = -7.0; sf_ac = -7.0; cm_ac = -7.0; sfvent = -7.0; lfvent = -7.0
    ep_pv = -7.0; qgr3 = -7.0; tgr3 = -7.0; draingr = -7.0
    lp = -7.0; lb = -7.0; hgt = 0.0; mh = -7.0; stdh = -7.0
    cmcr = -7.0; tgr2 = -7.0
    drelr = -7.0; drelb = -7.0; drelg = -7.0
    flxhumr = -7.0; flxhumb = -7.0; flxhumg = -7.0
    trl = -7.0; tbl = -7.0; tgl = -7.0; tgrl = -7.0; smr = -7.0
    lf = -7.0; hi = -7.0
    a_u = -7.0; a_v = -7.0; a_t = -7.0; a_q = -7.0; a_e = -7.0
    b_u = -7.0; b_v = -7.0; b_t = -7.0; b_q = -7.0; b_e = -7.0
    vl = -7.0; dlg = -7.0; sf = -7.0; dl_u = -7.0
    trb = -7.0; tw1 = -7.0; tw2 = -7.0; tgb = -7.0; tlev = -7.0; qlev = -7.0
    tw1lev = -7.0; tw2lev = -7.0; tglev = -7.0; tflev = -7.0
    sfwin1 = -7.0; sfwin2 = -7.0; sfw1 = -7.0; sfw2 = -7.0; sfr = -7.0
    sfg = -7.0; t_pv = -7.0; trv = -7.0; qr4 = -7.0; drain = -7.0
    sfrv = -7.0; lfrv = -7.0; dgr = -7.0; dg = -7.0; lfr = -7.0; lfg = -7.0
    utype = -7

    call urban_var_init(ISURBAN=isurban, TSURFACE0_URB=tsk, &
      TLAYER0_URB=tslb, TDEEP0_URB=tmn, IVGTYP=ivgtyp, &
      ims=1, ime=ncol, jms=1, jme=1, kms=kms, kme=kme, &
      num_soil_layers=nsoil, &
      LCZ_1=lcz(1), LCZ_2=lcz(2), LCZ_3=lcz(3), LCZ_4=lcz(4), &
      LCZ_5=lcz(5), LCZ_6=lcz(6), LCZ_7=lcz(7), LCZ_8=lcz(8), &
      LCZ_9=lcz(9), LCZ_10=lcz(10), LCZ_11=lcz(11), &
      restart=.false., sf_urban_physics=sf_urban_physics, &
      XXXR_URB2D=xxxr, XXXB_URB2D=xxxb, XXXG_URB2D=xxxg, XXXC_URB2D=xxxc, &
      TR_URB2D=tr, TB_URB2D=tb, TG_URB2D=tg, TC_URB2D=tc, QC_URB2D=qc, &
      TRL_URB3D=trl, TBL_URB3D=tbl, TGL_URB3D=tgl, &
      SH_URB2D=sh, LH_URB2D=lh, G_URB2D=g, RN_URB2D=rn, TS_URB2D=ts, &
      num_urban_ndm=ndm, urban_map_zrd=zrd, urban_map_zwd=zwd, &
      urban_map_gd=gd, urban_map_zd=zd, urban_map_zdf=zdf, &
      urban_map_bd=bd, urban_map_wd=wd, urban_map_gbd=gbd, &
      urban_map_fbd=fbd, urban_map_zgrd=zgrd, num_urban_hi=num_urban_hi, &
      TRB_URB4D=trb, TW1_URB4D=tw1, TW2_URB4D=tw2, TGB_URB4D=tgb, &
      TLEV_URB3D=tlev, QLEV_URB3D=qlev, TW1LEV_URB3D=tw1lev, &
      TW2LEV_URB3D=tw2lev, TGLEV_URB3D=tglev, TFLEV_URB3D=tflev, &
      SF_AC_URB3D=sf_ac, LF_AC_URB3D=lf_ac, CM_AC_URB3D=cm_ac, &
      SFVENT_URB3D=sfvent, LFVENT_URB3D=lfvent, &
      SFWIN1_URB3D=sfwin1, SFWIN2_URB3D=sfwin2, &
      SFW1_URB3D=sfw1, SFW2_URB3D=sfw2, SFR_URB3D=sfr, SFG_URB3D=sfg, &
      EP_PV_URB3D=ep_pv, T_PV_URB3D=t_pv, &
      TRV_URB4D=trv, QR_URB4D=qr4, QGR_URB3D=qgr3, TGR_URB3D=tgr3, &
      DRAIN_URB4D=drain, DRAINGR_URB3D=draingr, SFRV_URB3D=sfrv, &
      LFRV_URB3D=lfrv, DGR_URB3D=dgr, DG_URB3D=dg, LFR_URB3D=lfr, &
      LFG_URB3D=lfg, SMOIS_URB=smois, &
      LP_URB2D=lp, HI_URB2D=hi, LB_URB2D=lb, HGT_URB2D=hgt, MH_URB2D=mh, &
      STDH_URB2D=stdh, LF_URB2D=lf, &
      CMCR_URB2D=cmcr, TGR_URB2D=tgr2, TGRL_URB3D=tgrl, SMR_URB3D=smr, &
      DRELR_URB2D=drelr, DRELB_URB2D=drelb, DRELG_URB2D=drelg, &
      FLXHUMR_URB2D=flxhumr, FLXHUMB_URB2D=flxhumb, &
      FLXHUMG_URB2D=flxhumg, &
      A_U_BEP=a_u, A_V_BEP=a_v, A_T_BEP=a_t, A_Q_BEP=a_q, A_E_BEP=a_e, &
      B_U_BEP=b_u, B_V_BEP=b_v, B_T_BEP=b_t, B_Q_BEP=b_q, B_E_BEP=b_e, &
      DLG_BEP=dlg, DL_U_BEP=dl_u, SF_BEP=sf, VL_BEP=vl, &
      FRC_URB2D=frc, UTYPE_URB2D=utype, USE_WUDAPT_LCZ=use_wudapt_lcz)

    call oracle_put('frc_urb2d', frc)
    call oracle_put('utype_urb2d', utype)
    call oracle_put('xxxr_urb2d', xxxr); call oracle_put('xxxb_urb2d', xxxb)
    call oracle_put('xxxg_urb2d', xxxg); call oracle_put('xxxc_urb2d', xxxc)
    call oracle_put('tr_urb2d', tr); call oracle_put('tb_urb2d', tb)
    call oracle_put('tg_urb2d', tg); call oracle_put('tc_urb2d', tc)
    call oracle_put('qc_urb2d', qc)
    call oracle_put('sh_urb2d', sh); call oracle_put('lh_urb2d', lh)
    call oracle_put('g_urb2d', g); call oracle_put('rn_urb2d', rn)
    call oracle_put('ts_urb2d', ts)
    call oracle_put('trl_urb3d', trl); call oracle_put('tbl_urb3d', tbl)
    call oracle_put('tgl_urb3d', tgl); call oracle_put('tgrl_urb3d', tgrl)
    call oracle_put('smr_urb3d', smr)
    call oracle_put('cmcr_urb2d', cmcr); call oracle_put('tgr_urb2d', tgr2)
    call oracle_put('drelr_urb2d', drelr); call oracle_put('drelb_urb2d', drelb)
    call oracle_put('drelg_urb2d', drelg)
    call oracle_put('flxhumr_urb2d', flxhumr)
    call oracle_put('flxhumb_urb2d', flxhumb)
    call oracle_put('flxhumg_urb2d', flxhumg)
    call oracle_put('lp_urb2d', lp); call oracle_put('lb_urb2d', lb)
    call oracle_put('hgt_urb2d', hgt); call oracle_put('mh_urb2d', mh)
    call oracle_put('stdh_urb2d', stdh); call oracle_put('lf_urb2d', lf)
    call oracle_put('hi_urb2d', hi)
    call oracle_put('trb_urb4d', trb); call oracle_put('tw1_urb4d', tw1)
    call oracle_put('tw2_urb4d', tw2); call oracle_put('tgb_urb4d', tgb)
    call oracle_put('tlev_urb3d', tlev); call oracle_put('qlev_urb3d', qlev)
    call oracle_put('tw1lev_urb3d', tw1lev)
    call oracle_put('tw2lev_urb3d', tw2lev)
    call oracle_put('tglev_urb3d', tglev); call oracle_put('tflev_urb3d', tflev)
    call oracle_put('sf_ac_urb3d', sf_ac); call oracle_put('lf_ac_urb3d', lf_ac)
    call oracle_put('cm_ac_urb3d', cm_ac)
    call oracle_put('sfvent_urb3d', sfvent)
    call oracle_put('lfvent_urb3d', lfvent)
    call oracle_put('sfwin1_urb3d', sfwin1)
    call oracle_put('sfwin2_urb3d', sfwin2)
    call oracle_put('sfw1_urb3d', sfw1); call oracle_put('sfw2_urb3d', sfw2)
    call oracle_put('sfr_urb3d', sfr); call oracle_put('sfg_urb3d', sfg)
    call oracle_put('ep_pv_urb3d', ep_pv); call oracle_put('t_pv_urb3d', t_pv)
    call oracle_put('trv_urb4d', trv); call oracle_put('qr_urb4d', qr4)
    call oracle_put('qgr_urb3d', qgr3); call oracle_put('tgr_urb3d', tgr3)
    call oracle_put('drain_urb4d', drain)
    call oracle_put('draingr_urb3d', draingr)
    call oracle_put('sfrv_urb3d', sfrv); call oracle_put('lfrv_urb3d', lfrv)
    call oracle_put('dgr_urb3d', dgr); call oracle_put('dg_urb3d', dg)
    call oracle_put('lfr_urb3d', lfr); call oracle_put('lfg_urb3d', lfg)
    call oracle_put('a_u_bep', a_u); call oracle_put('a_v_bep', a_v)
    call oracle_put('a_t_bep', a_t); call oracle_put('a_q_bep', a_q)
    call oracle_put('a_e_bep', a_e); call oracle_put('b_u_bep', b_u)
    call oracle_put('b_v_bep', b_v); call oracle_put('b_t_bep', b_t)
    call oracle_put('b_q_bep', b_q); call oracle_put('b_e_bep', b_e)
    call oracle_put('dlg_bep', dlg); call oracle_put('dl_u_bep', dl_u)
    call oracle_put('sf_bep', sf); call oracle_put('vl_bep', vl)
  end subroutine one_case

  subroutine put_tables(dzr, dzb, dzg)
    real, intent(in) :: dzr(:), dzb(:), dzg(:)
    call oracle_put('ICATE', icate)
    call oracle_put('DZR', dzr); call oracle_put('DZB', dzb)
    call oracle_put('DZG', dzg)
    call oracle_put('ZR_TBL', zr_tbl(1:icate))
    call oracle_put('SIGMA_ZED_TBL', sigma_zed_tbl(1:icate))
    call oracle_put('Z0C_TBL', z0c_tbl(1:icate))
    call oracle_put('Z0HC_TBL', z0hc_tbl(1:icate))
    call oracle_put('ZDC_TBL', zdc_tbl(1:icate))
    call oracle_put('SVF_TBL', svf_tbl(1:icate))
    call oracle_put('R_TBL', r_tbl(1:icate))
    call oracle_put('RW_TBL', rw_tbl(1:icate))
    call oracle_put('HGT_TBL', hgt_tbl(1:icate))
    call oracle_put('AH_TBL', ah_tbl(1:icate))
    call oracle_put('ALH_TBL', alh_tbl(1:icate))
    call oracle_put('BETR_TBL', betr_tbl(1:icate))
    call oracle_put('BETB_TBL', betb_tbl(1:icate))
    call oracle_put('BETG_TBL', betg_tbl(1:icate))
    call oracle_put('FRC_URB_TBL', frc_urb_tbl(1:icate))
    call oracle_put('COP_TBL', cop_tbl(1:icate))
    call oracle_put('BLDAC_FRC_TBL', bldac_frc_tbl(1:icate))
    call oracle_put('COOLED_FRC_TBL', cooled_frc_tbl(1:icate))
    call oracle_put('PWIN_TBL', pwin_tbl(1:icate))
    call oracle_put('BETA_TBL', beta_tbl(1:icate))
    call oracle_put('SW_COND_TBL', sw_cond_tbl(1:icate))
    call oracle_put('TIME_ON_TBL', time_on_tbl(1:icate))
    call oracle_put('TIME_OFF_TBL', time_off_tbl(1:icate))
    call oracle_put('TARGTEMP_TBL', targtemp_tbl(1:icate))
    call oracle_put('GAPTEMP_TBL', gaptemp_tbl(1:icate))
    call oracle_put('TARGHUM_TBL', targhum_tbl(1:icate))
    call oracle_put('GAPHUM_TBL', gaphum_tbl(1:icate))
    call oracle_put('PERFLO_TBL', perflo_tbl(1:icate))
    call oracle_put('PV_FRAC_ROOF_TBL', pv_frac_roof_tbl(1:icate))
    call oracle_put('GR_FRAC_ROOF_TBL', gr_frac_roof_tbl(1:icate))
    call oracle_put('GR_FLAG_TBL', gr_flag_tbl)
    call oracle_put('GR_TYPE_TBL', gr_type_tbl)
    call oracle_put('IRHO_TBL', irho_tbl)
    call oracle_put('HSESF_TBL', hsesf_tbl(1:icate))
    call oracle_put('CAPR_TBL', capr_tbl(1:icate))
    call oracle_put('CAPB_TBL', capb_tbl(1:icate))
    call oracle_put('CAPG_TBL', capg_tbl(1:icate))
    call oracle_put('AKSR_TBL', aksr_tbl(1:icate))
    call oracle_put('AKSB_TBL', aksb_tbl(1:icate))
    call oracle_put('AKSG_TBL', aksg_tbl(1:icate))
    call oracle_put('ALBR_TBL', albr_tbl(1:icate))
    call oracle_put('ALBB_TBL', albb_tbl(1:icate))
    call oracle_put('ALBG_TBL', albg_tbl(1:icate))
    call oracle_put('EPSR_TBL', epsr_tbl(1:icate))
    call oracle_put('EPSB_TBL', epsb_tbl(1:icate))
    call oracle_put('EPSG_TBL', epsg_tbl(1:icate))
    call oracle_put('Z0R_TBL', z0r_tbl(1:icate))
    call oracle_put('Z0B_TBL', z0b_tbl(1:icate))
    call oracle_put('Z0G_TBL', z0g_tbl(1:icate))
    call oracle_put('Z0HB_TBL', z0hb_tbl(1:icate))
    call oracle_put('Z0HG_TBL', z0hg_tbl(1:icate))
    call oracle_put('TRLEND_TBL', trlend_tbl(1:icate))
    call oracle_put('TBLEND_TBL', tblend_tbl(1:icate))
    call oracle_put('TGLEND_TBL', tglend_tbl(1:icate))
    call oracle_put('AKANDA_URBAN_TBL', akanda_urban_tbl(1:icate))
    call oracle_put('NUMDIR_TBL', numdir_tbl(1:icate))
    call oracle_put('STREET_DIRECTION_TBL', street_direction_tbl(:, 1:icate))
    call oracle_put('STREET_WIDTH_TBL', street_width_tbl(:, 1:icate))
    call oracle_put('BUILDING_WIDTH_TBL', building_width_tbl(:, 1:icate))
    call oracle_put('NUMHGT_TBL', numhgt_tbl(1:icate))
    call oracle_put('HEIGHT_BIN_TBL', height_bin_tbl(:, 1:icate))
    call oracle_put('HPERCENT_BIN_TBL', hpercent_bin_tbl(:, 1:icate))
    call oracle_put('BOUNDR_DATA', boundr_data)
    call oracle_put('BOUNDB_DATA', boundb_data)
    call oracle_put('BOUNDG_DATA', boundg_data)
    call oracle_put('CH_SCHEME_DATA', ch_scheme_data)
    call oracle_put('TS_SCHEME_DATA', ts_scheme_data)
    call oracle_put('AHOPTION', ahoption)
    call oracle_put('AHDIUPRF', ahdiuprf)
    call oracle_put('HSEQUIP_TBL', hsequip_tbl)
    call oracle_put('IMP_SCHEME', imp_scheme)
    call oracle_put('IRI_SCHEME', iri_scheme)
    call oracle_put('ALHOPTION', alhoption)
    call oracle_put('GROPTION', groption)
    call oracle_put('FGR', fgr)
    call oracle_put('OASIS', oasis)
    call oracle_put('DZGR', dzgr)
    call oracle_put('ALHSEASON', alhseason)
    call oracle_put('ALHDIUPRF', alhdiuprf)
    call oracle_put('PORIMP', porimp)
    call oracle_put('DENGIMP', dengimp)
  end subroutine put_tables

end program run_urban_init_oracle
