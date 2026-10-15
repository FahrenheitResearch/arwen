! run_bep_bem_regime.F90 -- WRF v4.6.1 BEP+BEM (sf_urban_physics = 3) regime
! column oracle.
!
! Drives the byte-unmodified phys/module_sf_bep_bem.F (BEP_BEM) and
! phys/module_sf_bem.F exactly as module_sf_noahdrv.F does around its BEP_BEM
! call, after module_sf_urban.F's own urban_param_init (the table read) and
! urban_var_init (the option-3 initial state), over every weather regime of one
! season (REGIMES.csv) crossed with every urban class of the table, plus eight
! edge columns, for NSTEPS consecutive calls.  Every BEP_BEM input and output
! is dumped, together with every module_sf_urban table array BEP_BEM reads.
!
! usage: run_bep_bem_regime OUTDIR USE_WUDAPT_LCZ SEASON NSTEPS DT REGIMES_CSV [FULL]
!   SEASON is S (declination +0.40 rad, julian day 190) or W (-0.40, day 15).
!   URBPARM.TBL (USE_WUDAPT_LCZ=0) or URBPARM_LCZ.TBL (=1) must be in the
!   working directory, as WRF requires.  FULL=1 also dumps the whole state
!   before every call (stepN/in/...), for a replay diagnosis.
!
! Output: OUTDIR/data.bin (native little-endian float32/int32, concatenated)
! and OUTDIR/manifest.txt, one line per array:
!   <name> <f4|i4> <ndim> <d1> <d2> <d3> <d4> <offset_in_elements>
! The array names and layout are those of tools/urban_wrf471_oracle/
! run_bep_bem.F90, so one reader serves both.
!
! Nothing in this file computes a quantity the port is graded on: the columns
! are driver INPUTS.  Their solar geometry is made self-consistent (cosz from
! latitude, declination and hour angle) only so the inputs are plausible.

module bem_regime_dump
  implicit none
  integer, parameter :: ubin = 21, uman = 22
  integer(8) :: offset = 0
contains
  subroutine dump_open(outdir)
    character(len=*), intent(in) :: outdir
    open(ubin, file=trim(outdir)//'/data.bin', access='stream', &
         form='unformatted', status='replace')
    open(uman, file=trim(outdir)//'/manifest.txt', status='replace')
    offset = 0
  end subroutine dump_open

  subroutine dump_close()
    close(ubin)
    close(uman)
  end subroutine dump_close

  subroutine put_r(name, a, n, nd, d1, d2, d3, d4)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n, nd, d1, d2, d3, d4
    real, intent(in) :: a(n)
    write(ubin) a
    write(uman, '(A,1X,A,6(1X,I0))') trim(name), 'f4', nd, d1, d2, d3, d4, offset
    offset = offset + n
  end subroutine put_r

  subroutine put_i(name, a, n, nd, d1, d2, d3, d4)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n, nd, d1, d2, d3, d4
    integer, intent(in) :: a(n)
    write(ubin) a
    write(uman, '(A,1X,A,6(1X,I0))') trim(name), 'i4', nd, d1, d2, d3, d4, offset
    offset = offset + n
  end subroutine put_i

  subroutine put_rs(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    real :: a(1)
    a(1) = x
    call put_r(name, a, 1, 1, 1, 1, 1, 1)
  end subroutine put_rs

  subroutine put_is(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    integer :: a(1)
    a(1) = x
    call put_i(name, a, 1, 1, 1, 1, 1, 1)
  end subroutine put_is
end module bem_regime_dump


program run_bep_bem_regime
  use module_sf_urban
  use module_sf_bep_bem
  use module_bep_bem_helper, only: nurbm
  use bem_regime_dump
  implicit none

  integer, parameter :: nz = 30, nsoil = 4, nedge = 8, maxreg = 32
  integer, parameter :: n_ndm = 2, n_nz = 18, n_ng = 10, n_nwr = 10
  integer, parameter :: n_nf = 10, n_ngb = 10, n_nbui = 15, n_ngr = 10
  integer, parameter :: num_urban_hi = 15
  integer, parameter :: map_zrd = n_ndm * n_nwr * n_nz
  integer, parameter :: map_zwd = n_ndm * n_nwr * n_nz * n_nbui
  integer, parameter :: map_gd = n_ndm * n_ng
  integer, parameter :: map_zd = n_ndm * n_nz * n_nbui
  integer, parameter :: map_zdf = n_ndm * n_nz
  integer, parameter :: map_bd = n_nz * n_nbui
  integer, parameter :: map_wd = n_ndm * n_nz * n_nbui
  integer, parameter :: map_gbd = n_ndm * n_ngb * n_nbui
  integer, parameter :: map_fbd = n_ndm * (n_nz - 1) * n_nf * n_nbui
  integer, parameter :: map_zgrd = n_ndm * n_ngr * n_nz
  integer, parameter :: isurban = 13
  integer, parameter :: lcz(11) = (/ 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61 /)
  real, parameter :: frc_cycle(6) = (/ 0.0, 0.5, 0.9, 0.99, 0.3, 0.75 /)

  integer :: ncol, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  integer :: ids, ide, jds, jde, kds, kde

  character(len=512) :: outdir, arg, csv
  character(len=1) :: season
  integer :: use_wudapt_lcz, nsteps, istep, i, k, nreg, ncls, ir, ic, full
  real :: dt, gmt, declin
  integer :: julday
  logical :: restart

  ! regime table (REGIMES.csv rows of this season)
  character(len=96) :: rname(maxreg)
  real :: rv(14, maxreg)

  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil)

  real, allocatable, dimension(:, :) :: tsk, tmn
  real, allocatable, dimension(:, :, :) :: tslb, smois
  integer, allocatable :: ivgtyp(:, :), utype_urb2d(:, :)

  real, allocatable, dimension(:, :) :: xxxr, xxxb, xxxg, xxxc, tr, tb, tg, tc, qc
  real, allocatable, dimension(:, :) :: sh, lh, g, rn, ts, cmcr, tgr2d
  real, allocatable, dimension(:, :) :: drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  real, allocatable, dimension(:, :, :) :: trl, tbl, tgl, tgrl, smr
  real, allocatable, dimension(:, :) :: lp_urb2d, lb_urb2d, hgt_urb2d, mh_urb2d, stdh_urb2d
  real, allocatable :: lf_urb2d(:, :, :), hi_urb2d(:, :, :)
  real, allocatable :: frc_urb2d(:, :)

  real, allocatable, dimension(:, :, :) :: trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d
  real, allocatable, dimension(:, :, :) :: tlev_urb3d, qlev_urb3d, tw1lev_urb3d, tw2lev_urb3d
  real, allocatable, dimension(:, :, :) :: tglev_urb3d, tflev_urb3d
  real, allocatable, dimension(:, :) :: sf_ac_urb3d, lf_ac_urb3d, cm_ac_urb3d
  real, allocatable, dimension(:, :) :: sfvent_urb3d, lfvent_urb3d
  real, allocatable, dimension(:, :, :) :: sfwin1_urb3d, sfwin2_urb3d, sfw1_urb3d, sfw2_urb3d
  real, allocatable, dimension(:, :, :) :: sfr_urb3d, sfg_urb3d, t_pv_urb3d
  real, allocatable, dimension(:, :) :: ep_pv_urb3d, qgr_urb3d, tgr_urb3d, draingr_urb3d
  real, allocatable, dimension(:, :, :) :: trv_urb4d, qr_urb4d, drain_urb4d, sfrv_urb3d
  real, allocatable, dimension(:, :, :) :: lfrv_urb3d, dgr_urb3d, dg_urb3d, lfr_urb3d, lfg_urb3d

  real, allocatable, dimension(:, :, :) :: a_u, a_v, a_t, a_q, a_e, b_u, b_v, b_t, b_e, b_q
  real, allocatable, dimension(:, :, :) :: dlg, dl_u, sf, vl

  real, allocatable, dimension(:, :, :) :: dz8w, u_phy, v_phy, th_phy, rho, p_phy, qv_phy
  real, allocatable, dimension(:, :) :: swdown, glw, swddir, swddif, rainbl
  real, allocatable, dimension(:, :) :: xlat, xlong, cosz, omg
  real, allocatable, dimension(:, :) :: rl_up, rs_abs, emiss, grdflx_urb

  ! per-column design
  integer, allocatable :: cls(:), reg(:), edge(:), morph(:)
  real, allocatable :: frc_in(:), dz1(:)

  call get_command_argument(1, outdir)
  call get_command_argument(2, arg)
  read(arg, *) use_wudapt_lcz
  call get_command_argument(3, season)
  call get_command_argument(4, arg)
  read(arg, *) nsteps
  call get_command_argument(5, arg)
  read(arg, *) dt
  call get_command_argument(6, csv)
  full = 0
  if (command_argument_count() >= 7) then
    call get_command_argument(7, arg)
    read(arg, *) full
  end if

  if (season == 'S') then
    declin = 0.40
    julday = 190
  else if (season == 'W') then
    declin = -0.40
    julday = 15
  else
    write(*, *) 'SEASON must be S or W'
    error stop 2
  end if
  gmt = 0.0
  restart = .false.
  call read_regimes()

  if (use_wudapt_lcz == 0) then
    nurbm = 3
    ncls = 3
  else
    nurbm = 11
    ncls = 11
  end if

  dzr = 0.0
  dzb = 0.0
  dzg = 0.0
  call urban_param_init(dzr, dzb, dzg, nsoil, 3, use_wudapt_lcz, .false.)

  ! ---- column design --------------------------------------------------------
  ncol = nreg * ncls + nedge
  ims = 1; ime = ncol; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol; jts = 1; jte = 1; kts = 1; kte = nz
  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  call allocate_all()

  i = 0
  do ir = 1, nreg
    do ic = 1, ncls
      i = i + 1
      reg(i) = ir
      cls(i) = ic
      edge(i) = 0
      morph(i) = 0
      frc_in(i) = frc_cycle(mod(ir + ic, 6) + 1)
      dz1(i) = rv(12, ir)
    end do
  end do
  ! edge columns: 1 rural land, 2 open water, 3 FRC 0.01, 4 FRC 1.0,
  ! 5 gridded morphology, 6 gridded tall towers up to the 15th 5 m bin,
  ! 7 first level 40 m (the whole canopy inside level 1), 8 first level 2 m.
  do k = 1, nedge
    i = i + 1
    edge(i) = k
    morph(i) = 0
    reg(i) = mod(k - 1, nreg) + 1
    dz1(i) = rv(12, reg(i))
    select case (k)
    case (1)
      cls(i) = 0; frc_in(i) = 0.0
    case (2)
      cls(i) = -1; frc_in(i) = 0.0
    case (3)
      cls(i) = 1; frc_in(i) = 0.01
    case (4)
      cls(i) = ncls; frc_in(i) = 1.0
    case (5)
      cls(i) = 2; frc_in(i) = 0.7; morph(i) = 1
    case (6)
      cls(i) = 1; frc_in(i) = 0.85; morph(i) = 2
    case (7)
      cls(i) = 2; frc_in(i) = 0.6; dz1(i) = 40.0
    case (8)
      cls(i) = min(3, ncls); frc_in(i) = 0.8; dz1(i) = 2.0
    end select
  end do

  do i = its, ite
    ir = reg(i)
    if (cls(i) == 0) then
      ivgtyp(i, 1) = 10
    else if (cls(i) == -1) then
      ivgtyp(i, 1) = 17
    else if (use_wudapt_lcz == 0 .and. cls(i) == 2) then
      ivgtyp(i, 1) = isurban
    else if (use_wudapt_lcz == 1 .and. cls(i) == 5) then
      ivgtyp(i, 1) = isurban
    else
      ivgtyp(i, 1) = lcz(cls(i))
    end if
    tsk(i, 1) = rv(1, ir)
    tmn(i, 1) = rv(13, ir)
    do k = 1, nsoil
      tslb(i, k, 1) = rv(1, ir) + (rv(13, ir) - rv(1, ir)) * real(k) / real(nsoil + 1)
      smois(i, k, 1) = rv(14, ir)
    end do
    frc_urb2d(i, 1) = frc_in(i)
    lp_urb2d(i, 1) = 0.0
    lb_urb2d(i, 1) = 0.0
    hgt_urb2d(i, 1) = 0.0
    hi_urb2d(i, :, 1) = 0.0
    if (morph(i) == 1) then
      hgt_urb2d(i, 1) = 17.5
      lp_urb2d(i, 1) = 0.45
      lb_urb2d(i, 1) = 1.30
      hi_urb2d(i, 1:6, 1) = (/ 0.05, 0.20, 0.35, 0.25, 0.10, 0.05 /)
    else if (morph(i) == 2) then
      hgt_urb2d(i, 1) = 43.35
      lp_urb2d(i, 1) = 0.25
      lb_urb2d(i, 1) = 2.60
      hi_urb2d(i, 1:15, 1) = (/ 0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.10, 0.10, &
                                0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02 /)
    end if
    mh_urb2d(i, 1) = 0.0
    stdh_urb2d(i, 1) = 0.0
    lf_urb2d(i, :, 1) = 0.0
  end do

  call urban_var_init(isurban, tsk, tslb, tmn, ivgtyp,                    &
       ims, ime, jms, jme, kms, kme, nsoil,                              &
       lcz(1), lcz(2), lcz(3), lcz(4), lcz(5),                           &
       lcz(6), lcz(7), lcz(8), lcz(9), lcz(10), lcz(11),                 &
       restart, 3,                                                       &
       xxxr, xxxb, xxxg, xxxc, tr, tb, tg, tc, qc,                       &
       trl, tbl, tgl, sh, lh, g, rn, ts,                                 &
       n_ndm, map_zrd, map_zwd, map_gd, map_zd, map_zdf, map_bd,         &
       map_wd, map_gbd, map_fbd, map_zgrd, num_urban_hi,                 &
       trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d,                       &
       tlev_urb3d, qlev_urb3d, tw1lev_urb3d, tw2lev_urb3d,               &
       tglev_urb3d, tflev_urb3d,                                         &
       sf_ac_urb3d, lf_ac_urb3d, cm_ac_urb3d, sfvent_urb3d, lfvent_urb3d,&
       sfwin1_urb3d, sfwin2_urb3d,                                       &
       sfw1_urb3d, sfw2_urb3d, sfr_urb3d, sfg_urb3d,                     &
       ep_pv_urb3d, t_pv_urb3d,                                          &
       trv_urb4d, qr_urb4d, qgr_urb3d, tgr_urb3d,                        &
       drain_urb4d, draingr_urb3d, sfrv_urb3d,                           &
       lfrv_urb3d, dgr_urb3d, dg_urb3d, lfr_urb3d, lfg_urb3d,            &
       smois,                                                            &
       lp_urb2d, hi_urb2d, lb_urb2d,                                     &
       hgt_urb2d, mh_urb2d, stdh_urb2d, lf_urb2d,                        &
       cmcr, tgr2d, tgrl, smr,                                           &
       drelr, drelb, drelg, flxhumr, flxhumb, flxhumg,                   &
       a_u, a_v, a_t, a_q, a_e, b_u, b_v, b_t, b_q, b_e, dlg, dl_u, sf, vl, &
       frc_urb2d, utype_urb2d, use_wudapt_lcz)

  call dump_open(outdir)
  call put_is('use_wudapt_lcz', use_wudapt_lcz)
  call put_is('nsteps', nsteps)
  call put_is('ncol', ncol)
  call put_is('nz', nz)
  call put_is('num_urban_hi', num_urban_hi)
  call put_is('nurbm', nurbm)
  call put_is('nreg', nreg)
  call put_is('ncls', ncls)
  call put_is('season_winter', merge(1, 0, season == 'W'))
  call put_i('col/regime', reg, ncol, 1, ncol, 1, 1, 1)
  call put_i('col/class', cls, ncol, 1, ncol, 1, 1, 1)
  call put_i('col/edge', edge, ncol, 1, ncol, 1, 1, 1)
  call put_i('col/morph', morph, ncol, 1, ncol, 1, 1, 1)
  call dump_tables()

  call put_i('init/ivgtyp', ivgtyp, ncol, 2, ncol, 1, 1, 1)
  call put_r('init/tsk', tsk, ncol, 2, ncol, 1, 1, 1)
  call put_r('init/tslb', tslb, ncol*nsoil, 3, ncol, nsoil, 1, 1)
  call put_r('init/smois', smois, ncol*nsoil, 3, ncol, nsoil, 1, 1)
  call put_r('init/tmn', tmn, ncol, 2, ncol, 1, 1, 1)
  call put_r('init/frc_urb2d_in', frc_in, ncol, 2, ncol, 1, 1, 1)
  call dump_state('state0')

  do istep = 1, nsteps
    call set_atmosphere(istep)

    ! module_sf_noahdrv.F: the driver zeroes these before every BEP_BEM
    ! call, on every column.
    do i = its, ite
      emiss(i, 1) = 0.
      rl_up(i, 1) = 0.
      rs_abs(i, 1) = 0.
      grdflx_urb(i, 1) = 0.
      b_q(i, kts:kte, 1) = 0.
    end do

    write(arg, '(A,I0)') 'step', istep
    call dump_inputs(trim(arg))
    if (full == 1) call dump_state(trim(arg)//'/in')

    call BEP_BEM(frc_urb2d, utype_urb2d, istep, dz8w, dt, u_phy, v_phy,  &
         th_phy, rho, p_phy, swdown, glw,                                &
         gmt, julday, xlong, xlat, declin, cosz, omg,                    &
         n_ndm, map_zrd, map_zwd, map_gd,                                &
         map_zd, map_zdf, map_bd, map_wd,                                &
         map_gbd, map_fbd, map_zgrd, num_urban_hi,                       &
         trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d,                     &
         tlev_urb3d, qlev_urb3d, tw1lev_urb3d, tw2lev_urb3d,             &
         tglev_urb3d, tflev_urb3d, sf_ac_urb3d, lf_ac_urb3d,             &
         cm_ac_urb3d, sfvent_urb3d, lfvent_urb3d,                        &
         sfwin1_urb3d, sfwin2_urb3d,                                     &
         sfw1_urb3d, sfw2_urb3d, sfr_urb3d, sfg_urb3d,                   &
         ep_pv_urb3d, t_pv_urb3d,                                        &
         trv_urb4d, qr_urb4d, qgr_urb3d, tgr_urb3d,                      &
         drain_urb4d, draingr_urb3d, sfrv_urb3d,                         &
         lfrv_urb3d, dgr_urb3d, dg_urb3d, lfr_urb3d, lfg_urb3d,          &
         rainbl, swddir, swddif,                                         &
         lp_urb2d, hi_urb2d, lb_urb2d, hgt_urb2d,                        &
         a_u, a_v, a_t, a_e, b_u, b_v,                                   &
         b_t, b_e, b_q, dlg, dl_u, sf, vl,                               &
         rl_up, rs_abs, emiss, grdflx_urb, qv_phy,                       &
         ids, ide, jds, jde, kds, kde,                                   &
         ims, ime, jms, jme, kms, kme,                                   &
         its, ite, jts, jte, kts, kte)

    call dump_outputs(trim(arg))
    if (istep == nsteps) call dump_state('final')
  end do
  call dump_close()

contains

  subroutine read_regimes()
    character(len=1) :: s
    character(len=96) :: name
    integer :: id, ios
    real :: v(14)
    open(31, file=trim(csv), status='old', action='read')
    read(31, *)                      ! header
    nreg = 0
    do
      read(31, *, iostat=ios) s, id, name, v
      if (ios /= 0) exit
      if (s /= season) cycle
      nreg = nreg + 1
      if (nreg > maxreg .or. id /= nreg) then
        write(*, *) 'REGIMES.csv: season ', season, ' ids must run 1..n'
        error stop 3
      end if
      rname(nreg) = name
      rv(:, nreg) = v
    end do
    close(31)
    if (nreg == 0) error stop 4
  end subroutine read_regimes

  subroutine set_atmosphere(istep)
    integer, intent(in) :: istep
    real :: zc, thb, ah, lat_r, cz, dirfrac, tmin
    real, parameter :: rcp = 0.2857
    ! the slow drift of the forcing goes with elapsed time (minutes), not
    ! with the call count, so a run at a short step drifts no faster
    tmin = real(istep - 1) * dt / 60.0
    do i = its, ite
      ir = reg(i)
      do k = kts, kte
        dz8w(i, k, 1) = dz1(i) * 1.12**(k - 1)
      end do
      dz8w(i, kte + 1, 1) = dz8w(i, kte, 1)
      ! potential temperature of the first-level air: the regime's air
      ! temperature (surface minus dth) referred to 1000 hPa
      thb = (rv(1, ir) - rv(2, ir)) * (1.e5 / rv(9, ir))**rcp
      zc = 0.0
      do k = kts, kte
        zc = zc + 0.5 * dz8w(i, k, 1)
        if (k > kts) zc = zc + 0.5 * dz8w(i, k - 1, 1)
        th_phy(i, k, 1) = thb + 1.e-3 * rv(3, ir) * zc + 0.05 * tmin
        p_phy(i, k, 1) = rv(9, ir) * exp(-zc / 8000.0)
        rho(i, k, 1) = p_phy(i, k, 1) / (287.0 * th_phy(i, k, 1) * (p_phy(i, k, 1) / 1.e5)**rcp)
        u_phy(i, k, 1) = rv(4, ir) * log(1.0 + zc / 2.0) / log(1.0 + 100.0 / 2.0) + 0.1 * (tmin + 1.0)
        v_phy(i, k, 1) = 0.4 * rv(4, ir) * log(1.0 + zc / 2.0) / log(1.0 + 100.0 / 2.0) - 0.2
        qv_phy(i, k, 1) = rv(5, ir) * exp(-zc / 2500.0)
      end do
      do k = kte + 1, kme
        th_phy(i, k, 1) = th_phy(i, kte, 1)
        p_phy(i, k, 1) = p_phy(i, kte, 1)
        rho(i, k, 1) = rho(i, kte, 1)
        u_phy(i, k, 1) = u_phy(i, kte, 1)
        v_phy(i, k, 1) = v_phy(i, kte, 1)
        qv_phy(i, k, 1) = qv_phy(i, kte, 1)
      end do
      xlat(i, 1) = rv(8, ir)
      xlong(i, 1) = -120.0 + 2.0 * real(mod(i, 90))
      ah = (rv(7, ir) + real(istep - 1) * dt / 3600.0 - 12.0) * 15.0 * 3.14159265 / 180.0
      omg(i, 1) = ah
      lat_r = rv(8, ir) * 3.14159265 / 180.0
      cz = sin(lat_r) * sin(declin) + cos(lat_r) * cos(declin) * cos(ah)
      cosz(i, 1) = cz
      if (cz > 0.0) then
        swdown(i, 1) = 1000.0 * cz * rv(10, ir)
      else
        swdown(i, 1) = 0.0
      end if
      dirfrac = 0.85 * rv(10, ir)
      swddir(i, 1) = dirfrac * swdown(i, 1)
      swddif(i, 1) = swdown(i, 1) - swddir(i, 1)
      glw(i, 1) = rv(11, ir) + 0.5 * tmin
      rainbl(i, 1) = rv(6, ir)
    end do
  end subroutine set_atmosphere

  subroutine allocate_all()
    allocate(tsk(ims:ime, jms:jme), tmn(ims:ime, jms:jme))
    allocate(tslb(ims:ime, nsoil, jms:jme), smois(ims:ime, nsoil, jms:jme))
    allocate(ivgtyp(ims:ime, jms:jme), utype_urb2d(ims:ime, jms:jme))
    allocate(xxxr(ims:ime, jms:jme), xxxb(ims:ime, jms:jme), xxxg(ims:ime, jms:jme))
    allocate(xxxc(ims:ime, jms:jme), tr(ims:ime, jms:jme), tb(ims:ime, jms:jme))
    allocate(tg(ims:ime, jms:jme), tc(ims:ime, jms:jme), qc(ims:ime, jms:jme))
    allocate(sh(ims:ime, jms:jme), lh(ims:ime, jms:jme), g(ims:ime, jms:jme))
    allocate(rn(ims:ime, jms:jme), ts(ims:ime, jms:jme), cmcr(ims:ime, jms:jme))
    allocate(tgr2d(ims:ime, jms:jme), drelr(ims:ime, jms:jme), drelb(ims:ime, jms:jme))
    allocate(drelg(ims:ime, jms:jme), flxhumr(ims:ime, jms:jme), flxhumb(ims:ime, jms:jme))
    allocate(flxhumg(ims:ime, jms:jme))
    allocate(trl(ims:ime, nsoil, jms:jme), tbl(ims:ime, nsoil, jms:jme))
    allocate(tgl(ims:ime, nsoil, jms:jme), tgrl(ims:ime, nsoil, jms:jme))
    allocate(smr(ims:ime, nsoil, jms:jme))
    allocate(lp_urb2d(ims:ime, jms:jme), lb_urb2d(ims:ime, jms:jme))
    allocate(hgt_urb2d(ims:ime, jms:jme), mh_urb2d(ims:ime, jms:jme))
    allocate(stdh_urb2d(ims:ime, jms:jme), lf_urb2d(ims:ime, 4, jms:jme))
    allocate(hi_urb2d(ims:ime, num_urban_hi, jms:jme), frc_urb2d(ims:ime, jms:jme))
    allocate(trb_urb4d(ims:ime, map_zrd, jms:jme))
    allocate(tw1_urb4d(ims:ime, map_zwd, jms:jme), tw2_urb4d(ims:ime, map_zwd, jms:jme))
    allocate(tgb_urb4d(ims:ime, map_gd, jms:jme))
    allocate(tlev_urb3d(ims:ime, map_bd, jms:jme), qlev_urb3d(ims:ime, map_bd, jms:jme))
    allocate(tw1lev_urb3d(ims:ime, map_wd, jms:jme), tw2lev_urb3d(ims:ime, map_wd, jms:jme))
    allocate(tglev_urb3d(ims:ime, map_gbd, jms:jme), tflev_urb3d(ims:ime, map_fbd, jms:jme))
    allocate(sf_ac_urb3d(ims:ime, jms:jme), lf_ac_urb3d(ims:ime, jms:jme))
    allocate(cm_ac_urb3d(ims:ime, jms:jme), sfvent_urb3d(ims:ime, jms:jme))
    allocate(lfvent_urb3d(ims:ime, jms:jme))
    allocate(sfwin1_urb3d(ims:ime, map_wd, jms:jme), sfwin2_urb3d(ims:ime, map_wd, jms:jme))
    allocate(sfw1_urb3d(ims:ime, map_zd, jms:jme), sfw2_urb3d(ims:ime, map_zd, jms:jme))
    allocate(sfr_urb3d(ims:ime, map_zdf, jms:jme), sfg_urb3d(ims:ime, n_ndm, jms:jme))
    allocate(ep_pv_urb3d(ims:ime, jms:jme), t_pv_urb3d(ims:ime, map_zdf, jms:jme))
    allocate(trv_urb4d(ims:ime, map_zgrd, jms:jme), qr_urb4d(ims:ime, map_zgrd, jms:jme))
    allocate(qgr_urb3d(ims:ime, jms:jme), tgr_urb3d(ims:ime, jms:jme))
    allocate(drain_urb4d(ims:ime, map_zdf, jms:jme), draingr_urb3d(ims:ime, jms:jme))
    allocate(sfrv_urb3d(ims:ime, map_zdf, jms:jme), lfrv_urb3d(ims:ime, map_zdf, jms:jme))
    allocate(dgr_urb3d(ims:ime, map_zdf, jms:jme), dg_urb3d(ims:ime, n_ndm, jms:jme))
    allocate(lfr_urb3d(ims:ime, map_zdf, jms:jme), lfg_urb3d(ims:ime, n_ndm, jms:jme))
    allocate(a_u(ims:ime, kms:kme, jms:jme), a_v(ims:ime, kms:kme, jms:jme))
    allocate(a_t(ims:ime, kms:kme, jms:jme), a_q(ims:ime, kms:kme, jms:jme))
    allocate(a_e(ims:ime, kms:kme, jms:jme), b_u(ims:ime, kms:kme, jms:jme))
    allocate(b_v(ims:ime, kms:kme, jms:jme), b_t(ims:ime, kms:kme, jms:jme))
    allocate(b_e(ims:ime, kms:kme, jms:jme), b_q(ims:ime, kms:kme, jms:jme))
    allocate(dlg(ims:ime, kms:kme, jms:jme), dl_u(ims:ime, kms:kme, jms:jme))
    allocate(sf(ims:ime, kms:kme, jms:jme), vl(ims:ime, kms:kme, jms:jme))
    allocate(dz8w(ims:ime, kms:kme, jms:jme), u_phy(ims:ime, kms:kme, jms:jme))
    allocate(v_phy(ims:ime, kms:kme, jms:jme), th_phy(ims:ime, kms:kme, jms:jme))
    allocate(rho(ims:ime, kms:kme, jms:jme), p_phy(ims:ime, kms:kme, jms:jme))
    allocate(qv_phy(ims:ime, kms:kme, jms:jme))
    allocate(swdown(ims:ime, jms:jme), glw(ims:ime, jms:jme), swddir(ims:ime, jms:jme))
    allocate(swddif(ims:ime, jms:jme), rainbl(ims:ime, jms:jme))
    allocate(xlat(ims:ime, jms:jme), xlong(ims:ime, jms:jme))
    allocate(cosz(ims:ime, jms:jme), omg(ims:ime, jms:jme))
    allocate(rl_up(its:ite, jts:jte), rs_abs(its:ite, jts:jte))
    allocate(emiss(its:ite, jts:jte), grdflx_urb(its:ite, jts:jte))
    allocate(cls(ncol), reg(ncol), edge(ncol), morph(ncol), frc_in(ncol), dz1(ncol))
  end subroutine allocate_all

  subroutine dump_tables()
    integer :: nc
    nc = icate
    call put_is('tbl/icate', icate)
    call put_r('tbl/capr_tbl', capr_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/capb_tbl', capb_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/capg_tbl', capg_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/aksr_tbl', aksr_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/aksb_tbl', aksb_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/aksg_tbl', aksg_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/albr_tbl', albr_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/albb_tbl', albb_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/albg_tbl', albg_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/epsr_tbl', epsr_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/epsb_tbl', epsb_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/epsg_tbl', epsg_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/z0r_tbl', z0r_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/z0g_tbl', z0g_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/trlend_tbl', trlend_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/tblend_tbl', tblend_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/tglend_tbl', tglend_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/frc_urb_tbl', frc_urb_tbl, nc, 1, nc, 1, 1, 1)
    call put_i('tbl/numdir_tbl', numdir_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/street_direction_tbl', street_direction_tbl, maxdirs*nc, 2, maxdirs, nc, 1, 1)
    call put_r('tbl/street_width_tbl', street_width_tbl, maxdirs*nc, 2, maxdirs, nc, 1, 1)
    call put_r('tbl/building_width_tbl', building_width_tbl, maxdirs*nc, 2, maxdirs, nc, 1, 1)
    call put_i('tbl/numhgt_tbl', numhgt_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/height_bin_tbl', height_bin_tbl, maxhgts*nc, 2, maxhgts, nc, 1, 1)
    call put_r('tbl/hpercent_bin_tbl', hpercent_bin_tbl, maxhgts*nc, 2, maxhgts, nc, 1, 1)
    call put_r('tbl/cop_tbl', cop_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/bldac_frc_tbl', bldac_frc_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/cooled_frc_tbl', cooled_frc_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/pwin_tbl', pwin_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/beta_tbl', beta_tbl, nc, 1, nc, 1, 1, 1)
    call put_i('tbl/sw_cond_tbl', sw_cond_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/time_on_tbl', time_on_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/time_off_tbl', time_off_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/targtemp_tbl', targtemp_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/gaptemp_tbl', gaptemp_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/targhum_tbl', targhum_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/gaphum_tbl', gaphum_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/perflo_tbl', perflo_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/pv_frac_roof_tbl', pv_frac_roof_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/gr_frac_roof_tbl', gr_frac_roof_tbl, nc, 1, nc, 1, 1, 1)
    call put_is('tbl/gr_flag_tbl', gr_flag_tbl)
    call put_is('tbl/gr_type_tbl', gr_type_tbl)
    call put_r('tbl/irho_tbl', irho_tbl, 24, 1, 24, 1, 1, 1)
    call put_r('tbl/hsesf_tbl', hsesf_tbl, nc, 1, nc, 1, 1, 1)
    call put_r('tbl/hsequip_tbl', hsequip_tbl, 24, 1, 24, 1, 1, 1)
  end subroutine dump_tables

  subroutine dump_state(tag)
    character(len=*), intent(in) :: tag
    integer :: nk
    nk = kme - kms + 1
    call put_r(tag//'/frc_urb2d', frc_urb2d, ncol, 2, ncol, 1, 1, 1)
    call put_i(tag//'/utype_urb2d', utype_urb2d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/lp_urb2d', lp_urb2d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/lb_urb2d', lb_urb2d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/hgt_urb2d', hgt_urb2d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/hi_urb2d', hi_urb2d, ncol*num_urban_hi, 3, ncol, num_urban_hi, 1, 1)
    call put_r(tag//'/trb_urb4d', trb_urb4d, ncol*map_zrd, 3, ncol, map_zrd, 1, 1)
    call put_r(tag//'/tw1_urb4d', tw1_urb4d, ncol*map_zwd, 3, ncol, map_zwd, 1, 1)
    call put_r(tag//'/tw2_urb4d', tw2_urb4d, ncol*map_zwd, 3, ncol, map_zwd, 1, 1)
    call put_r(tag//'/tgb_urb4d', tgb_urb4d, ncol*map_gd, 3, ncol, map_gd, 1, 1)
    call put_r(tag//'/tlev_urb3d', tlev_urb3d, ncol*map_bd, 3, ncol, map_bd, 1, 1)
    call put_r(tag//'/qlev_urb3d', qlev_urb3d, ncol*map_bd, 3, ncol, map_bd, 1, 1)
    call put_r(tag//'/tw1lev_urb3d', tw1lev_urb3d, ncol*map_wd, 3, ncol, map_wd, 1, 1)
    call put_r(tag//'/tw2lev_urb3d', tw2lev_urb3d, ncol*map_wd, 3, ncol, map_wd, 1, 1)
    call put_r(tag//'/tglev_urb3d', tglev_urb3d, ncol*map_gbd, 3, ncol, map_gbd, 1, 1)
    call put_r(tag//'/tflev_urb3d', tflev_urb3d, ncol*map_fbd, 3, ncol, map_fbd, 1, 1)
    call put_r(tag//'/sf_ac_urb3d', sf_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/lf_ac_urb3d', lf_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/cm_ac_urb3d', cm_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/sfvent_urb3d', sfvent_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/lfvent_urb3d', lfvent_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/sfwin1_urb3d', sfwin1_urb3d, ncol*map_wd, 3, ncol, map_wd, 1, 1)
    call put_r(tag//'/sfwin2_urb3d', sfwin2_urb3d, ncol*map_wd, 3, ncol, map_wd, 1, 1)
    call put_r(tag//'/sfw1_urb3d', sfw1_urb3d, ncol*map_zd, 3, ncol, map_zd, 1, 1)
    call put_r(tag//'/sfw2_urb3d', sfw2_urb3d, ncol*map_zd, 3, ncol, map_zd, 1, 1)
    call put_r(tag//'/sfr_urb3d', sfr_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/sfg_urb3d', sfg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
    call put_r(tag//'/ep_pv_urb3d', ep_pv_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/t_pv_urb3d', t_pv_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/trv_urb4d', trv_urb4d, ncol*map_zgrd, 3, ncol, map_zgrd, 1, 1)
    call put_r(tag//'/qr_urb4d', qr_urb4d, ncol*map_zgrd, 3, ncol, map_zgrd, 1, 1)
    call put_r(tag//'/qgr_urb3d', qgr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/tgr_urb3d', tgr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/drain_urb4d', drain_urb4d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/draingr_urb3d', draingr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/sfrv_urb3d', sfrv_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/lfrv_urb3d', lfrv_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/dgr_urb3d', dgr_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/dg_urb3d', dg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
    call put_r(tag//'/lfr_urb3d', lfr_urb3d, ncol*map_zdf, 3, ncol, map_zdf, 1, 1)
    call put_r(tag//'/lfg_urb3d', lfg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
    call put_r(tag//'/a_u_bep', a_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/a_v_bep', a_v, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/a_t_bep', a_t, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/a_q_bep', a_q, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/a_e_bep', a_e, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/b_u_bep', b_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/b_v_bep', b_v, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/b_t_bep', b_t, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/b_q_bep', b_q, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/b_e_bep', b_e, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/dlg_bep', dlg, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/dl_u_bep', dl_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/sf_bep', sf, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/vl_bep', vl, ncol*nk, 3, ncol, nk, 1, 1)
  end subroutine dump_state

  subroutine dump_inputs(tag)
    character(len=*), intent(in) :: tag
    integer :: nk
    nk = kme - kms + 1
    call put_r(tag//'/dz8w', dz8w, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/u_phy', u_phy, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/v_phy', v_phy, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/th_phy', th_phy, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/rho', rho, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/p_phy', p_phy, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/qv_phy', qv_phy, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/swdown', swdown, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/glw', glw, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/swddir', swddir, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/swddif', swddif, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/rainbl', rainbl, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/cosz_urb2d', cosz, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/omg_urb2d', omg, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/xlat', xlat, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/xlong', xlong, ncol, 2, ncol, 1, 1, 1)
    call put_rs(tag//'/gmt', gmt)
    call put_is(tag//'/julday', julday)
    call put_rs(tag//'/declin_urb', declin)
    call put_rs(tag//'/dt', dt)
    call put_is(tag//'/itimestep', istep)
  end subroutine dump_inputs

  subroutine dump_outputs(tag)
    character(len=*), intent(in) :: tag
    integer :: nk
    nk = kme - kms + 1
    call put_r(tag//'/out/a_u', a_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/a_v', a_v, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/a_t', a_t, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/a_e', a_e, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/b_u', b_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/b_v', b_v, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/b_t', b_t, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/b_e', b_e, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/b_q', b_q, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/dlg', dlg, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/dl_u', dl_u, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/sf', sf, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/vl', vl, ncol*nk, 3, ncol, nk, 1, 1)
    call put_r(tag//'/out/rl_up', rl_up, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/rs_abs', rs_abs, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/emiss', emiss, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/grdflx_urb', grdflx_urb, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/sf_ac_urb3d', sf_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/lf_ac_urb3d', lf_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/cm_ac_urb3d', cm_ac_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/sfvent_urb3d', sfvent_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/lfvent_urb3d', lfvent_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/ep_pv_urb3d', ep_pv_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/qgr_urb3d', qgr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/tgr_urb3d', tgr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/draingr_urb3d', draingr_urb3d, ncol, 2, ncol, 1, 1, 1)
    call put_r(tag//'/out/sfg_urb3d', sfg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
    call put_r(tag//'/out/dg_urb3d', dg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
    call put_r(tag//'/out/lfg_urb3d', lfg_urb3d, ncol*n_ndm, 3, ncol, n_ndm, 1, 1)
  end subroutine dump_outputs

end program run_bep_bem_regime
