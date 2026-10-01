program BEP_PROGRAM_NAME
  ! BEP column oracle, WRF v4.7.1 phys/module_sf_bep.F, byte-unmodified.
  !
  ! Drives SUBROUTINE BEP exactly as module_sf_noahdrv.F:1603-1631 calls it
  ! (sf_urban_physics = 2), over a one-row "domain" whose columns are the
  ! fixture cases, for NSTEP consecutive steps.  Every input BEP reads and
  ! every word it writes is dumped per step, so the GPU port is graded from
  ! the SAME inputs each step (strict identity) and can also be chained.
  !
  ! Table set: "nlcd" = URBPARM.TBL (use_wudapt_lcz=0, nurbm=3,
  ! check_a_mundo.F:457-461), "lcz" = URBPARM_LCZ.TBL (=1, nurbm=11).
  ! BEP's own "first" block (init_para + icBEP, SAVEd) runs once per process,
  ! which is why each table set is its own program (run_bep_nlcd.F90,
  ! run_bep_lcz.F90 include this body with BEP_TABLE_SET defined); the
  ! output root is the first argument (oracle_io's convention).
  !
  ! Initial state = urban_var_init's BEP arm (module_sf_urban.F:2912-2930):
  ! TRB/TW1/TW2 = TBLEND_TBL(UTYPE), TGB = TLAYER0(1), SFW/SFR/SFG = 0; the
  ! fixture then perturbs the layer temperatures with a recorded, column-
  ! dependent profile so the conduction solves see non-uniform layers.
  !
  ! Output words that BEP must NOT touch (FRC_URB2D <= 0 columns, and every
  ! column's k > kte slots) are preset to a sentinel so the port is checked
  ! for leaving them alone.
  use module_bep_bem_helper, only: nurbm
  use module_sf_urban, only: urban_param_init, tblend_tbl, icate,          &
      capb_tbl, capr_tbl, capg_tbl, aksb_tbl, aksr_tbl, aksg_tbl,          &
      trlend_tbl, tglend_tbl, albb_tbl, albr_tbl, albg_tbl, epsb_tbl,      &
      epsr_tbl, epsg_tbl, z0r_tbl, z0g_tbl, numdir_tbl,                    &
      street_direction_tbl, street_width_tbl, building_width_tbl,          &
      numhgt_tbl, height_bin_tbl, hpercent_bin_tbl
  use module_sf_bep, only: bep
  use oracle_io
  implicit none

  integer, parameter :: nz = 30          ! mass levels (kte)
  integer, parameter :: nstep = 4
  integer, parameter :: ndm = 2, nz_um = 18, ng_u = 10, nwr_u = 10, nbui = 1
  integer, parameter :: num_urban_ndm = ndm
  integer, parameter :: urban_map_zrd = ndm * nwr_u * nz_um
  integer, parameter :: urban_map_zwd = ndm * nwr_u * nz_um * nbui
  integer, parameter :: urban_map_gd = ndm * ng_u
  integer, parameter :: urban_map_zd = ndm * nz_um * nbui
  integer, parameter :: urban_map_zdf = ndm * nz_um
  integer, parameter :: urban_map_bd = nz_um * nbui
  integer, parameter :: urban_map_wd = ndm * nz_um * nbui
  integer, parameter :: urban_map_gbd = ndm * 10 * nbui
  integer, parameter :: urban_map_fbd = ndm * (nz_um - 1) * 10 * nbui
  integer, parameter :: num_urban_hi = 15
  integer, parameter :: num_soil_layers = 4
  real, parameter :: sentinel = -777.25

  character(len=16) :: tset
  character(len=1024) :: outdir
  character(len=64) :: tag
  integer :: use_wudapt_lcz, ncol, icol, k, istep, l, itimestep, julday
  real :: dt, gmt, declin_urb
  real :: dzr(num_soil_layers), dzb(num_soil_layers), dzg(num_soil_layers)

  real, allocatable :: dz8w(:,:,:), u_phy(:,:,:), v_phy(:,:,:), th_phy(:,:,:)
  real, allocatable :: rho(:,:,:), p_phy(:,:,:)
  real, allocatable :: swdown(:,:), glw(:,:), xlong(:,:), xlat(:,:)
  real, allocatable :: cosz_urb2d(:,:), omg_urb2d(:,:), frc_urb2d(:,:)
  integer, allocatable :: utype_urb2d(:,:)
  real, allocatable :: lp_urb2d(:,:), lb_urb2d(:,:), hgt_urb2d(:,:)
  real, allocatable :: hi_urb2d(:,:,:)
  real, allocatable :: trb_urb4d(:,:,:), tw1_urb4d(:,:,:), tw2_urb4d(:,:,:)
  real, allocatable :: tgb_urb4d(:,:,:), sfw1_urb3d(:,:,:), sfw2_urb3d(:,:,:)
  real, allocatable :: sfr_urb3d(:,:,:), sfg_urb3d(:,:,:)
  real, allocatable :: a_u(:,:,:), a_v(:,:,:), a_t(:,:,:), a_e(:,:,:)
  real, allocatable :: b_u(:,:,:), b_v(:,:,:), b_t(:,:,:), b_e(:,:,:)
  real, allocatable :: b_q(:,:,:), dlg(:,:,:), dl_u(:,:,:), sf(:,:,:), vl(:,:,:)
  real, allocatable :: rl_up(:,:), rs_abs(:,:), emiss(:,:), grdflx_urb(:,:)
  real, allocatable :: tsoil1(:)
  integer, allocatable :: utype_in(:)

  tset = BEP_TABLE_SET
  call oracle_root(outdir)
  if (trim(tset) == 'nlcd') then
    use_wudapt_lcz = 0
    nurbm = 3
  else if (trim(tset) == 'lcz') then
    use_wudapt_lcz = 1
    nurbm = 11
  else
    error stop 2
  end if

  dzr = 0.; dzb = 0.; dzg = 0.
  call urban_param_init(dzr, dzb, dzg, num_soil_layers, 2, use_wudapt_lcz, .false.)

  call define_cases()

  allocate(dz8w(ncol, nz + 1, 1), u_phy(ncol, nz + 1, 1), v_phy(ncol, nz + 1, 1))
  allocate(th_phy(ncol, nz + 1, 1), rho(ncol, nz + 1, 1), p_phy(ncol, nz + 1, 1))
  allocate(swdown(ncol, 1), glw(ncol, 1), xlong(ncol, 1), xlat(ncol, 1))
  allocate(cosz_urb2d(ncol, 1), omg_urb2d(ncol, 1), frc_urb2d(ncol, 1))
  allocate(utype_urb2d(ncol, 1), lp_urb2d(ncol, 1), lb_urb2d(ncol, 1))
  allocate(hgt_urb2d(ncol, 1), hi_urb2d(ncol, num_urban_hi, 1))
  allocate(trb_urb4d(ncol, urban_map_zrd, 1), tw1_urb4d(ncol, urban_map_zwd, 1))
  allocate(tw2_urb4d(ncol, urban_map_zwd, 1), tgb_urb4d(ncol, urban_map_gd, 1))
  allocate(sfw1_urb3d(ncol, urban_map_zd, 1), sfw2_urb3d(ncol, urban_map_zd, 1))
  allocate(sfr_urb3d(ncol, urban_map_zdf, 1), sfg_urb3d(ncol, num_urban_ndm, 1))
  allocate(a_u(ncol, nz + 1, 1), a_v(ncol, nz + 1, 1), a_t(ncol, nz + 1, 1))
  allocate(a_e(ncol, nz + 1, 1), b_u(ncol, nz + 1, 1), b_v(ncol, nz + 1, 1))
  allocate(b_t(ncol, nz + 1, 1), b_e(ncol, nz + 1, 1), b_q(ncol, nz + 1, 1))
  allocate(dlg(ncol, nz + 1, 1), dl_u(ncol, nz + 1, 1), sf(ncol, nz + 1, 1))
  allocate(vl(ncol, nz + 1, 1))
  allocate(rl_up(ncol, 1), rs_abs(ncol, 1), emiss(ncol, 1), grdflx_urb(ncol, 1))

  call build_columns()

  dt = 20.
  gmt = 17.5
  julday = 197
  declin_urb = 0.3752
  xlong = -97.
  xlat = 35.

  call oracle_open('steps')
  call oracle_put('meta_nz', real(nz))
  call oracle_put('meta_ncol', real(ncol))
  call oracle_put('meta_nstep', real(nstep))
  call oracle_put('meta_use_wudapt_lcz', real(use_wudapt_lcz))
  call oracle_put('meta_dt', dt)
  call oracle_put('meta_declin_urb', declin_urb)
  ! The module words BEP's init_para reads (module_sf_bep.F:3057-3098), as
  ! urban_param_init left them -- i.e. after its unit conversions.  The
  ! port's table packer is fed exactly these, so the column gate grades BEP
  ! and not the table reader (infra's urban_tables is graded separately).
  call oracle_put('tbl_ICATE', [icate])
  call oracle_put('tbl_CAPB_TBL', capb_tbl); call oracle_put('tbl_CAPR_TBL', capr_tbl)
  call oracle_put('tbl_CAPG_TBL', capg_tbl); call oracle_put('tbl_AKSB_TBL', aksb_tbl)
  call oracle_put('tbl_AKSR_TBL', aksr_tbl); call oracle_put('tbl_AKSG_TBL', aksg_tbl)
  call oracle_put('tbl_TBLEND_TBL', tblend_tbl); call oracle_put('tbl_TRLEND_TBL', trlend_tbl)
  call oracle_put('tbl_TGLEND_TBL', tglend_tbl); call oracle_put('tbl_ALBB_TBL', albb_tbl)
  call oracle_put('tbl_ALBR_TBL', albr_tbl); call oracle_put('tbl_ALBG_TBL', albg_tbl)
  call oracle_put('tbl_EPSB_TBL', epsb_tbl); call oracle_put('tbl_EPSR_TBL', epsr_tbl)
  call oracle_put('tbl_EPSG_TBL', epsg_tbl); call oracle_put('tbl_Z0R_TBL', z0r_tbl)
  call oracle_put('tbl_Z0G_TBL', z0g_tbl); call oracle_put('tbl_NUMDIR_TBL', numdir_tbl)
  call oracle_put('tbl_STREET_DIRECTION_TBL', street_direction_tbl)
  call oracle_put('tbl_STREET_WIDTH_TBL', street_width_tbl)
  call oracle_put('tbl_BUILDING_WIDTH_TBL', building_width_tbl)
  call oracle_put('tbl_NUMHGT_TBL', numhgt_tbl)
  call oracle_put('tbl_HEIGHT_BIN_TBL', height_bin_tbl)
  call oracle_put('tbl_HPERCENT_BIN_TBL', hpercent_bin_tbl)
  call oracle_put('dz8w', dz8w(:, :, 1))
  call oracle_put('u_phy', u_phy(:, :, 1))
  call oracle_put('v_phy', v_phy(:, :, 1))
  call oracle_put('th_phy', th_phy(:, :, 1))
  call oracle_put('rho', rho(:, :, 1))
  call oracle_put('p_phy', p_phy(:, :, 1))
  call oracle_put('swdown', swdown(:, 1))
  call oracle_put('glw', glw(:, 1))
  call oracle_put('cosz_urb2d', cosz_urb2d(:, 1))
  call oracle_put('omg_urb2d', omg_urb2d(:, 1))
  call oracle_put('frc_urb2d', frc_urb2d(:, 1))
  call oracle_put('utype_urb2d', utype_urb2d(:, 1))
  call oracle_put('lp_urb2d', lp_urb2d(:, 1))
  call oracle_put('lb_urb2d', lb_urb2d(:, 1))
  call oracle_put('hgt_urb2d', hgt_urb2d(:, 1))
  call oracle_put('hi_urb2d', hi_urb2d(:, :, 1))

  do istep = 1, nstep
    itimestep = istep
    write(tag, '(A,I0,A)') 's', istep, '_in_'
    call put_state(trim(tag))

    ! noahdrv.F:1607-1611 zeroes only these before the call.
    do icol = 1, ncol
      emiss(icol, 1) = 0.
      rl_up(icol, 1) = 0.
      rs_abs(icol, 1) = 0.
      grdflx_urb(icol, 1) = 0.
      b_q(icol, 1:nz, 1) = 0.
    end do
    ! Everything else starts at a sentinel so untouched words are visible.
    a_u = sentinel; a_v = sentinel; a_t = sentinel; a_e = sentinel
    b_u = sentinel; b_v = sentinel; b_t = sentinel; b_e = sentinel
    dlg = sentinel; dl_u = sentinel; sf = sentinel; vl = sentinel
    b_q(:, nz + 1, 1) = sentinel

    call bep(frc_urb2d, utype_urb2d, itimestep, dz8w, dt, u_phy, v_phy,     &
             th_phy, rho, p_phy, swdown, glw,                               &
             gmt, julday, xlong, xlat, declin_urb, cosz_urb2d, omg_urb2d,   &
             num_urban_ndm, urban_map_zrd, urban_map_zwd, urban_map_gd,     &
             urban_map_zd, urban_map_zdf, urban_map_bd, urban_map_wd,       &
             urban_map_gbd, urban_map_fbd, num_urban_hi,                    &
             trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d,                    &
             sfw1_urb3d, sfw2_urb3d, sfr_urb3d, sfg_urb3d,                  &
             lp_urb2d, hi_urb2d, lb_urb2d, hgt_urb2d,                       &
             a_u, a_v, a_t, a_e, b_u, b_v, b_t, b_e, b_q, dlg, dl_u, sf, vl, &
             rl_up, rs_abs, emiss, grdflx_urb,                              &
             1, ncol + 1, 1, 2, 1, nz + 1,                                  &
             1, ncol, 1, 1, 1, nz + 1,                                      &
             1, ncol, 1, 1, 1, nz)

    write(tag, '(A,I0,A)') 's', istep, '_out_'
    call put_state(trim(tag))
    call oracle_put(trim(tag)//'a_u', a_u(:, :, 1))
    call oracle_put(trim(tag)//'a_v', a_v(:, :, 1))
    call oracle_put(trim(tag)//'a_t', a_t(:, :, 1))
    call oracle_put(trim(tag)//'a_e', a_e(:, :, 1))
    call oracle_put(trim(tag)//'b_u', b_u(:, :, 1))
    call oracle_put(trim(tag)//'b_v', b_v(:, :, 1))
    call oracle_put(trim(tag)//'b_t', b_t(:, :, 1))
    call oracle_put(trim(tag)//'b_e', b_e(:, :, 1))
    call oracle_put(trim(tag)//'b_q', b_q(:, :, 1))
    call oracle_put(trim(tag)//'dlg', dlg(:, :, 1))
    call oracle_put(trim(tag)//'dl_u', dl_u(:, :, 1))
    call oracle_put(trim(tag)//'sf', sf(:, :, 1))
    call oracle_put(trim(tag)//'vl', vl(:, :, 1))
    call oracle_put(trim(tag)//'rl_up', rl_up(:, 1))
    call oracle_put(trim(tag)//'rs_abs', rs_abs(:, 1))
    call oracle_put(trim(tag)//'emiss', emiss(:, 1))
    call oracle_put(trim(tag)//'grdflx_urb', grdflx_urb(:, 1))
  end do
  call oracle_close()
  write(*, '(A,A,A,I0,A,I0,A)') 'bep oracle ', trim(tset), ': ', ncol, &
    ' columns x ', nstep, ' steps written'

contains

  subroutine put_state(prefix)
    character(len=*), intent(in) :: prefix
    call oracle_put(prefix//'trb_urb4d', trb_urb4d(:, :, 1))
    call oracle_put(prefix//'tw1_urb4d', tw1_urb4d(:, :, 1))
    call oracle_put(prefix//'tw2_urb4d', tw2_urb4d(:, :, 1))
    call oracle_put(prefix//'tgb_urb4d', tgb_urb4d(:, :, 1))
    call oracle_put(prefix//'sfw1_urb3d', sfw1_urb3d(:, :, 1))
    call oracle_put(prefix//'sfw2_urb3d', sfw2_urb3d(:, :, 1))
    call oracle_put(prefix//'sfr_urb3d', sfr_urb3d(:, :, 1))
    call oracle_put(prefix//'sfg_urb3d', sfg_urb3d(:, :, 1))
  end subroutine put_state

  include 'bep_cases.inc'

end program BEP_PROGRAM_NAME
