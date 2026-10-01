program BEP_PROGRAM_NAME
  ! sf_urban_physics = 2 "after the LSM", end to end, WRF v4.7.1:
  !   zeroing (module_sf_noahdrv.F:1606-1612 / noahmpdrv.F:3610-3618)
  !   -> BEP (module_sf_bep.F, byte-unmodified)
  !   -> the frc weighting / rural merge / skin-temperature and flux blend,
  ! once with Noah's block and once with Noah-MP's, for NSTEP consecutive
  ! steps, each chain carrying its own coupled a/b/sf/vl arrays forward
  ! (BEP leaves non-urban columns' arrays as the previous couple left them,
  ! so the second step exercises the stale-value path).
  !
  ! The two coupling blocks are NOT re-typed: they are INCLUDEd from files
  ! cut out of the pinned sources by line range,
  !   couple_noahdrv_1679_1776.inc              = noahdrv.F:1679-1776
  !   couple_noahmpdrv_3363_3372_3689_3776.inc  = noahmpdrv.F:3363-3372
  !                                               + noahmpdrv.F:3689-3776
  ! and PROVENANCE.md records the sed commands and their sha256.
  !
  ! The rural inputs are what each LSM leaves behind for its block: Noah's
  ! HFX_RURAL/QFX_RURAL/... are SHEAT/ETA_KINEMATIC/... (noahdrv.F:1225-1273),
  ! the same words it writes to HFX/QFX/...; Noah-MP's block reads HFX/QFX/...
  ! itself (noahmpdrv.F:3363-3372).  The fixture supplies them per column.
  use module_bep_bem_helper, only: nurbm
  use module_sf_urban, only: urban_param_init, tblend_tbl
  use module_sf_bep, only: bep
  use module_model_constants, only: cp, xlv, rcp
  use oracle_io
  implicit none

  integer, parameter :: nz = 30
  integer, parameter :: nstep = 2
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

  character(len=16) :: tset
  character(len=1024) :: outdir
  character(len=64) :: tag
  integer :: use_wudapt_lcz, ncol, icol, istep, itimestep, julday, ilsm
  integer :: sf_urban_physics
  integer :: its, ite, jts, jte, kts, kte, i, j, k
  real :: dt, gmt, declin_urb, sigma_sb
  real :: umom, vmom, rl_up_rural, rl_up_tot, rs_abs_tot
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
  real, allocatable :: tsoil1(:)
  integer, allocatable :: utype_in(:)
  ! BEP state snapshot so both LSM chains advance BEP's own state once/step
  real, allocatable :: s_trb(:,:,:), s_tw1(:,:,:), s_tw2(:,:,:), s_tgb(:,:,:)
  real, allocatable :: s_sfw1(:,:,:), s_sfw2(:,:,:), s_sfr(:,:,:), s_sfg(:,:,:)

  ! PBL handoff, one set per LSM chain (last index)
  real, allocatable :: c_a_u(:,:,:,:), c_a_v(:,:,:,:), c_a_t(:,:,:,:), c_a_q(:,:,:,:)
  real, allocatable :: c_a_e(:,:,:,:), c_b_u(:,:,:,:), c_b_v(:,:,:,:), c_b_t(:,:,:,:)
  real, allocatable :: c_b_q(:,:,:,:), c_b_e(:,:,:,:), c_dlg(:,:,:,:), c_dl_u(:,:,:,:)
  real, allocatable :: c_sf(:,:,:,:), c_vl(:,:,:,:)
  ! working names exactly as the included WRF blocks spell them
  real, allocatable :: a_u_bep(:,:,:), a_v_bep(:,:,:), a_t_bep(:,:,:), a_q_bep(:,:,:)
  real, allocatable :: a_e_bep(:,:,:), b_u_bep(:,:,:), b_v_bep(:,:,:), b_t_bep(:,:,:)
  real, allocatable :: b_q_bep(:,:,:), b_e_bep(:,:,:), dlg_bep(:,:,:), dl_u_bep(:,:,:)
  real, allocatable :: sf_bep(:,:,:), vl_bep(:,:,:)
  real, allocatable :: rl_up_urb(:,:), rs_abs_urb(:,:), emiss_urb(:,:), grdflx_urb(:,:)
  real, allocatable :: umom_urb(:,:), vmom_urb(:,:), hfx_urb(:,:), qfx_urb(:,:)
  real, allocatable :: hfx_rural(:,:), qfx_rural(:,:), grdflx_rural(:,:)
  real, allocatable :: emiss_rural(:,:), tsk_rural(:,:), alb_rural(:,:)
  real, allocatable :: ust(:,:), tsk(:,:), hfx(:,:), qfx(:,:), lh(:,:), grdflx(:,:)
  real, allocatable :: albedo(:,:), emiss(:,:), ts_urb2d(:,:)
  real, allocatable :: sh_urb2d(:,:), lh_urb2d(:,:), g_urb2d(:,:), rn_urb2d(:,:)
  ! the LSM's rural result per column (inputs)
  real, allocatable :: l_ust(:), l_tsk(:), l_hfx(:), l_qfx(:), l_grdflx(:)
  real, allocatable :: l_albedo(:), l_emiss(:)
  ! surface-driver override inputs/outputs (module_surface_driver.F:3028-3032)
  real, allocatable :: psfc(:,:), qv_curr(:,:,:), t2(:,:), th2(:,:), q2(:,:)
  real, allocatable :: u10(:,:), v10(:,:)

  tset = BEP_TABLE_SET
  call oracle_root(outdir)
  if (trim(tset) == 'nlcd') then
    use_wudapt_lcz = 0; nurbm = 3
  else if (trim(tset) == 'lcz') then
    use_wudapt_lcz = 1; nurbm = 11
  else
    error stop 2
  end if
  sf_urban_physics = 2
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
  allocate(c_a_u(ncol, nz + 1, 1, 2), c_a_v(ncol, nz + 1, 1, 2), c_a_t(ncol, nz + 1, 1, 2))
  allocate(c_a_q(ncol, nz + 1, 1, 2), c_a_e(ncol, nz + 1, 1, 2), c_b_u(ncol, nz + 1, 1, 2))
  allocate(c_b_v(ncol, nz + 1, 1, 2), c_b_t(ncol, nz + 1, 1, 2), c_b_q(ncol, nz + 1, 1, 2))
  allocate(c_b_e(ncol, nz + 1, 1, 2), c_dlg(ncol, nz + 1, 1, 2), c_dl_u(ncol, nz + 1, 1, 2))
  allocate(c_sf(ncol, nz + 1, 1, 2), c_vl(ncol, nz + 1, 1, 2))
  allocate(a_u_bep(ncol, nz + 1, 1), a_v_bep(ncol, nz + 1, 1), a_t_bep(ncol, nz + 1, 1))
  allocate(a_q_bep(ncol, nz + 1, 1), a_e_bep(ncol, nz + 1, 1), b_u_bep(ncol, nz + 1, 1))
  allocate(b_v_bep(ncol, nz + 1, 1), b_t_bep(ncol, nz + 1, 1), b_q_bep(ncol, nz + 1, 1))
  allocate(b_e_bep(ncol, nz + 1, 1), dlg_bep(ncol, nz + 1, 1), dl_u_bep(ncol, nz + 1, 1))
  allocate(sf_bep(ncol, nz + 1, 1), vl_bep(ncol, nz + 1, 1))
  allocate(rl_up_urb(ncol, 1), rs_abs_urb(ncol, 1), emiss_urb(ncol, 1), grdflx_urb(ncol, 1))
  allocate(umom_urb(ncol, 1), vmom_urb(ncol, 1), hfx_urb(ncol, 1), qfx_urb(ncol, 1))
  allocate(hfx_rural(ncol, 1), qfx_rural(ncol, 1), grdflx_rural(ncol, 1))
  allocate(emiss_rural(ncol, 1), tsk_rural(ncol, 1), alb_rural(ncol, 1))
  allocate(ust(ncol, 1), tsk(ncol, 1), hfx(ncol, 1), qfx(ncol, 1), lh(ncol, 1))
  allocate(grdflx(ncol, 1), albedo(ncol, 1), emiss(ncol, 1), ts_urb2d(ncol, 1))
  allocate(sh_urb2d(ncol, 1), lh_urb2d(ncol, 1), g_urb2d(ncol, 1), rn_urb2d(ncol, 1))
  allocate(l_ust(ncol), l_tsk(ncol), l_hfx(ncol), l_qfx(ncol), l_grdflx(ncol))
  allocate(l_albedo(ncol), l_emiss(ncol))

  call build_columns()
  call build_rural()

  its = 1; ite = ncol; jts = 1; jte = 1; kts = 1; kte = nz
  dt = 20.
  gmt = 17.5
  julday = 197
  declin_urb = 0.3752
  xlong = -97.
  xlat = 35.
  sigma_sb = 5.67e-08           ! noahdrv.F:727

  ! urban_var_init (module_sf_urban.F:2974-2990): sf = vl = 1, rest 0.
  c_a_u = 0.; c_a_v = 0.; c_a_t = 0.; c_a_q = 0.; c_a_e = 0.
  c_b_u = 0.; c_b_v = 0.; c_b_t = 0.; c_b_q = 0.; c_b_e = 0.
  c_dlg = 0.; c_dl_u = 0.; c_sf = 1.; c_vl = 1.

  call oracle_open('steps')
  call oracle_put('meta_nz', real(nz))
  call oracle_put('meta_ncol', real(ncol))
  call oracle_put('meta_nstep', real(nstep))
  call oracle_put('meta_use_wudapt_lcz', real(use_wudapt_lcz))
  call oracle_put('meta_dt', dt)
  call oracle_put('meta_declin_urb', declin_urb)
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
  call oracle_put('lsm_ust', l_ust); call oracle_put('lsm_tsk', l_tsk)
  call oracle_put('lsm_hfx', l_hfx); call oracle_put('lsm_qfx', l_qfx)
  call oracle_put('lsm_grdflx', l_grdflx); call oracle_put('lsm_albedo', l_albedo)
  call oracle_put('lsm_emiss', l_emiss)

  do istep = 1, nstep
    itimestep = istep
    s_trb = trb_urb4d; s_tw1 = tw1_urb4d; s_tw2 = tw2_urb4d; s_tgb = tgb_urb4d
    s_sfw1 = sfw1_urb3d; s_sfw2 = sfw2_urb3d; s_sfr = sfr_urb3d; s_sfg = sfg_urb3d
    write(tag, '(A,I0,A)') 's', istep, '_in_'
    call put_state(trim(tag))
    do ilsm = 1, 2
      trb_urb4d = s_trb; tw1_urb4d = s_tw1; tw2_urb4d = s_tw2; tgb_urb4d = s_tgb
      sfw1_urb3d = s_sfw1; sfw2_urb3d = s_sfw2; sfr_urb3d = s_sfr; sfg_urb3d = s_sfg
      a_u_bep = c_a_u(:, :, :, ilsm); a_v_bep = c_a_v(:, :, :, ilsm)
      a_t_bep = c_a_t(:, :, :, ilsm); a_q_bep = c_a_q(:, :, :, ilsm)
      a_e_bep = c_a_e(:, :, :, ilsm); b_u_bep = c_b_u(:, :, :, ilsm)
      b_v_bep = c_b_v(:, :, :, ilsm); b_t_bep = c_b_t(:, :, :, ilsm)
      b_q_bep = c_b_q(:, :, :, ilsm); b_e_bep = c_b_e(:, :, :, ilsm)
      dlg_bep = c_dlg(:, :, :, ilsm); dl_u_bep = c_dl_u(:, :, :, ilsm)
      sf_bep = c_sf(:, :, :, ilsm); vl_bep = c_vl(:, :, :, ilsm)
      if (ilsm == 1) then
        tag = 'noah_'
      else
        tag = 'noahmp_'
      end if
      write(tag, '(A,A,I0,A)') trim(tag), 's', istep, '_'
      call put_pbl(trim(tag)//'pre_')

      ! The LSM's rural result for this step (identical for both LSMs, so
      ! the only difference between the chains is the block that runs).
      ust(:, 1) = l_ust; tsk(:, 1) = l_tsk; hfx(:, 1) = l_hfx
      qfx(:, 1) = l_qfx; lh(:, 1) = l_qfx * xlv; grdflx(:, 1) = l_grdflx
      albedo(:, 1) = l_albedo; emiss(:, 1) = l_emiss
      ts_urb2d = -1.; sh_urb2d = -1.; lh_urb2d = -1.; g_urb2d = -1.; rn_urb2d = -1.
      if (ilsm == 1) then
        ! Noah's rural words (noahdrv.F:1225-1273) are the words it wrote.
        hfx_rural = hfx; qfx_rural = qfx; grdflx_rural = grdflx
        emiss_rural = emiss; tsk_rural = tsk; alb_rural = albedo
      end if

      ! noahdrv.F:1606-1612 == noahmpdrv.F:3610-3618
      do j = jts, jte
        do i = its, ite
          emiss_urb(i, j) = 0.
          rl_up_urb(i, j) = 0.
          rs_abs_urb(i, j) = 0.
          grdflx_urb(i, j) = 0.
          b_q_bep(i, kts:kte, j) = 0.
        end do
      end do
      call bep(frc_urb2d, utype_urb2d, itimestep, dz8w, dt, u_phy, v_phy,     &
               th_phy, rho, p_phy, swdown, glw,                               &
               gmt, julday, xlong, xlat, declin_urb, cosz_urb2d, omg_urb2d,   &
               num_urban_ndm, urban_map_zrd, urban_map_zwd, urban_map_gd,     &
               urban_map_zd, urban_map_zdf, urban_map_bd, urban_map_wd,       &
               urban_map_gbd, urban_map_fbd, num_urban_hi,                    &
               trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d,                    &
               sfw1_urb3d, sfw2_urb3d, sfr_urb3d, sfg_urb3d,                  &
               lp_urb2d, hi_urb2d, lb_urb2d, hgt_urb2d,                       &
               a_u_bep, a_v_bep, a_t_bep, a_e_bep, b_u_bep, b_v_bep,          &
               b_t_bep, b_e_bep, b_q_bep, dlg_bep, dl_u_bep, sf_bep, vl_bep,  &
               rl_up_urb, rs_abs_urb, emiss_urb, grdflx_urb,                  &
               1, ncol + 1, 1, 2, 1, nz + 1,                                  &
               1, ncol, 1, 1, 1, nz + 1,                                      &
               1, ncol, 1, 1, 1, nz)
      call oracle_put(trim(tag)//'bep_rl_up_urb', rl_up_urb(:, 1))
      call oracle_put(trim(tag)//'bep_rs_abs_urb', rs_abs_urb(:, 1))
      call oracle_put(trim(tag)//'bep_emiss_urb', emiss_urb(:, 1))
      call oracle_put(trim(tag)//'bep_grdflx_urb', grdflx_urb(:, 1))
      call put_pbl(trim(tag)//'bep_')

      if (ilsm == 1) then
        call couple_noah()
      else
        call couple_noahmp()
      end if

      call put_pbl(trim(tag)//'out_')
      call oracle_put(trim(tag)//'out_ust', ust(:, 1))
      call oracle_put(trim(tag)//'out_tsk', tsk(:, 1))
      call oracle_put(trim(tag)//'out_hfx', hfx(:, 1))
      call oracle_put(trim(tag)//'out_qfx', qfx(:, 1))
      call oracle_put(trim(tag)//'out_lh', lh(:, 1))
      call oracle_put(trim(tag)//'out_grdflx', grdflx(:, 1))
      call oracle_put(trim(tag)//'out_albedo', albedo(:, 1))
      call oracle_put(trim(tag)//'out_emiss', emiss(:, 1))
      call oracle_put(trim(tag)//'out_ts_urb2d', ts_urb2d(:, 1))
      call oracle_put(trim(tag)//'out_sh_urb2d', sh_urb2d(:, 1))
      call oracle_put(trim(tag)//'out_lh_urb2d', lh_urb2d(:, 1))
      call oracle_put(trim(tag)//'out_g_urb2d', g_urb2d(:, 1))
      call oracle_put(trim(tag)//'out_rn_urb2d', rn_urb2d(:, 1))

      c_a_u(:, :, :, ilsm) = a_u_bep; c_a_v(:, :, :, ilsm) = a_v_bep
      c_a_t(:, :, :, ilsm) = a_t_bep; c_a_q(:, :, :, ilsm) = a_q_bep
      c_a_e(:, :, :, ilsm) = a_e_bep; c_b_u(:, :, :, ilsm) = b_u_bep
      c_b_v(:, :, :, ilsm) = b_v_bep; c_b_t(:, :, :, ilsm) = b_t_bep
      c_b_q(:, :, :, ilsm) = b_q_bep; c_b_e(:, :, :, ilsm) = b_e_bep
      c_dlg(:, :, :, ilsm) = dlg_bep; c_dl_u(:, :, :, ilsm) = dl_u_bep
      c_sf(:, :, :, ilsm) = sf_bep; c_vl(:, :, :, ilsm) = vl_bep
    end do
    write(tag, '(A,I0,A)') 's', istep, '_out_'
    call put_state(trim(tag))
  end do
  ! Options 2/3 surface diagnostics override, :3022-3035 (Noah) and
  ! :3408-3421 (Noah-MP) carry the same five lines; IVGTYP urban <=> UTYPE>0.
  allocate(psfc(ncol, 1), qv_curr(ncol, nz + 1, 1), t2(ncol, 1), th2(ncol, 1))
  allocate(q2(ncol, 1), u10(ncol, 1), v10(ncol, 1))
  do icol = 1, ncol
    psfc(icol, 1) = p_phy(icol, 1, 1) + 0.5 * rho(icol, 1, 1) * 9.81 * dz8w(icol, 1, 1)
    qv_curr(icol, :, 1) = 0.004 + 0.0003 * real(mod(icol, 9))
    t2(icol, 1) = -1.; th2(icol, 1) = -1.; q2(icol, 1) = -1.
    u10(icol, 1) = -1.; v10(icol, 1) = -1.
  end do
  do j = jts, jte
    do i = its, ite
      if (utype_urb2d(i, j) > 0) then
        include 'sfcdiag_surface_driver_3028_3032.inc'
      end if
    end do
  end do
  call oracle_put('sfcdiag_psfc', psfc(:, 1))
  call oracle_put('sfcdiag_qv', qv_curr(:, :, 1))
  call oracle_put('sfcdiag_t2', t2(:, 1)); call oracle_put('sfcdiag_th2', th2(:, 1))
  call oracle_put('sfcdiag_q2', q2(:, 1)); call oracle_put('sfcdiag_u10', u10(:, 1))
  call oracle_put('sfcdiag_v10', v10(:, 1))
  call oracle_close()
  write(*, '(A,A,A,I0,A,I0,A)') 'bep couple oracle ', trim(tset), ': ', ncol, &
    ' columns x ', nstep, ' steps x 2 LSM blocks written'

contains

  subroutine couple_noah()
    include 'couple_noahdrv_1679_1776.inc'
  end subroutine couple_noah

  subroutine couple_noahmp()
    include 'couple_noahmpdrv_3363_3372_3689_3776.inc'
  end subroutine couple_noahmp

  subroutine put_pbl(prefix)
    character(len=*), intent(in) :: prefix
    call oracle_put(prefix//'a_u_bep', a_u_bep(:, :, 1))
    call oracle_put(prefix//'a_v_bep', a_v_bep(:, :, 1))
    call oracle_put(prefix//'a_t_bep', a_t_bep(:, :, 1))
    call oracle_put(prefix//'a_q_bep', a_q_bep(:, :, 1))
    call oracle_put(prefix//'a_e_bep', a_e_bep(:, :, 1))
    call oracle_put(prefix//'b_u_bep', b_u_bep(:, :, 1))
    call oracle_put(prefix//'b_v_bep', b_v_bep(:, :, 1))
    call oracle_put(prefix//'b_t_bep', b_t_bep(:, :, 1))
    call oracle_put(prefix//'b_q_bep', b_q_bep(:, :, 1))
    call oracle_put(prefix//'b_e_bep', b_e_bep(:, :, 1))
    call oracle_put(prefix//'dlg_bep', dlg_bep(:, :, 1))
    call oracle_put(prefix//'dl_u_bep', dl_u_bep(:, :, 1))
    call oracle_put(prefix//'sf_bep', sf_bep(:, :, 1))
    call oracle_put(prefix//'vl_bep', vl_bep(:, :, 1))
  end subroutine put_pbl

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

  subroutine build_rural()
    ! What an LSM plausibly leaves on each column: skin temperature near the
    ! lowest air temperature, day/night-signed fluxes, grass-like albedo.
    do icol = 1, ncol
      l_tsk(icol) = th_phy(icol, 1, 1) + 0.25 * (swdown(icol, 1) / 100. - 2.)
      l_hfx(icol) = 0.28 * swdown(icol, 1) - 25. + 3. * real(mod(icol, 5))
      l_qfx(icol) = 1.2e-7 * swdown(icol, 1) + 2.0e-6
      l_grdflx(icol) = -0.08 * swdown(icol, 1) + 12.
      l_albedo(icol) = 0.16 + 0.01 * real(mod(icol, 4))
      l_emiss(icol) = 0.95 + 0.005 * real(mod(icol, 7))
      l_ust(icol) = 0.12 + 0.05 * sqrt(u_phy(icol, 1, 1)**2 + v_phy(icol, 1, 1)**2)
    end do
  end subroutine build_rural

  include 'bep_cases.inc'

end program BEP_PROGRAM_NAME
