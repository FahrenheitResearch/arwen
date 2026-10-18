program run_mynn_bl_driver_columns_gsd41_oracle
  ! Column oracle of the GSD MYNN v4.1 mynn_bl_driver of the operational
  ! WRF 3.9 fork (NOAA-EMC/HRRR tag v4.1.21,
  ! sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F, unmodified, or with
  ! the one :995 line build.sh patches) over the columns written by
  ! columns_gsd41.py (84 columns, 21 regimes x 4 variants).
  !
  ! The namelist identity, the one-column call per column, the between-step
  ! update and the carried state are those of run_driver_families_gsd41.F90:
  ! HRRR's hrrr_wrf.nl (mixing length 2, cloudpdf 2, EDMF 1 with momentum,
  ! no EDMF TKE, cloudmix 1, mixqt 0, mixscalars 0, icloud_bl 1,
  ! grav_settling 0, no SPP, no restart, no cycling, FLAG_QC and FLAG_QI
  ! true).  Step 1 is the cold start, steps 2..N warm; between steps
  ! x = x + delt*tend (species floored at 0), t3d = th*exner,
  ! rho = p/(287*t3d).  The surface layer's ust, hfx, qfx and 1/L are held.
  !
  ! The input carries a snow profile; the fork's driver has no snow
  ! argument, so it is read and never used.
  !
  ! Everything is float32 bits in a little-endian stream file (layout in
  ! tests/_mynn_columns_gsd41.py).  Only the starting columns and each
  ! step's outputs are written: every later step's input is the previous
  ! step's outputs and the update below, which the reader repeats.
  !
  ! Arguments: INPUT.bin OUTPUT.bin
  use module_bl_mynn, only: mynn_bl_driver
  implicit none

  integer, parameter :: nsca = 12, npro = 14
  integer, parameter :: magic_in = 1297698353   ! 0x4D594E31
  integer :: ncol, nz, nstep, magic, uin, uout, c, k, s, initflag
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  real :: delt, dx
  character(len=1024) :: in_path, out_path
  real, allocatable :: scal(:, :), prof(:, :, :)
  ! state carried across steps, (nz, field, column)
  real, allocatable :: pin(:, :, :), sin_(:, :, :), outp(:, :, :)
  real, allocatable :: pblh_c(:), pblh_o(:), maxmf_o(:)
  integer, allocatable :: kpbl_c(:), kpbl_o(:), ktop_o(:)

  real, allocatable, dimension(:, :, :) :: dz3, u, v, w, th, qv, qc, qi, &
      qnc, qni, qnwfa, qnifa, p, exner, rho, t3d, qke, qke_adv, tsq, qsq, &
      cov, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten,      &
      rqncblten, rqniblten, rqnwfablten, rqnifablten, exch_h, exch_m,     &
      el_pbl, dqke, qwt, qshear, qbuoy, qdiss, sh3d, qc_bl, cldfra_bl,    &
      edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc,               &
      pattern_spp_pbl, rthraten
  real, dimension(1, 1) :: znt, xland, ts, qsfc, qcg, ps, ust, ch, hfx,    &
      qfx, rmol, wspd, uoce, voce, vdfg, pblh, wstar, delta, maxmf,        &
      mf_at_base
  integer, dimension(1, 1) :: kpbl, nupdraft, ktop_plume, kbot_shallow

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_driver_columns_gsd41 INPUT.bin OUTPUT.bin'
    error stop 2
  end if
  open(newunit=uin, file=trim(in_path), access='stream', form='unformatted', &
      status='old', action='read', convert='little_endian')
  read(uin) magic, ncol, nz, nstep, delt
  if (magic /= magic_in) error stop 'bad input magic'
  allocate(scal(nsca, ncol), prof(nz, npro, ncol))
  read(uin) scal, prof
  close(uin)

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz
  allocate(dz3(1, nz, 1), u(1, nz, 1), v(1, nz, 1), w(1, nz, 1),          &
      th(1, nz, 1), qv(1, nz, 1), qc(1, nz, 1), qi(1, nz, 1),             &
      qnc(1, nz, 1), qni(1, nz, 1), qnwfa(1, nz, 1), qnifa(1, nz, 1),     &
      p(1, nz, 1), exner(1, nz, 1), rho(1, nz, 1), t3d(1, nz, 1),         &
      qke(1, nz, 1), qke_adv(1, nz, 1), tsq(1, nz, 1), qsq(1, nz, 1),     &
      cov(1, nz, 1), rublten(1, nz, 1), rvblten(1, nz, 1),                &
      rthblten(1, nz, 1), rqvblten(1, nz, 1), rqcblten(1, nz, 1),         &
      rqiblten(1, nz, 1), rqncblten(1, nz, 1), rqniblten(1, nz, 1),       &
      rqnwfablten(1, nz, 1), rqnifablten(1, nz, 1), exch_h(1, nz, 1),     &
      exch_m(1, nz, 1), el_pbl(1, nz, 1), dqke(1, nz, 1), qwt(1, nz, 1),  &
      qshear(1, nz, 1), qbuoy(1, nz, 1), qdiss(1, nz, 1), sh3d(1, nz, 1), &
      qc_bl(1, nz, 1), cldfra_bl(1, nz, 1), edmf_a(1, nz, 1),             &
      edmf_w(1, nz, 1), edmf_qt(1, nz, 1), edmf_thl(1, nz, 1),            &
      edmf_ent(1, nz, 1), edmf_qc(1, nz, 1), pattern_spp_pbl(1, nz, 1),   &
      rthraten(1, nz, 1))

  ! pin fields: u v th qv qc qi t3d rho; sin fields: qke tsq qsq cov el sh
  ! qc_bl cldfra_bl
  allocate(pin(nz, 8, ncol), sin_(nz, 8, ncol), outp(nz, 16, ncol))
  allocate(pblh_c(ncol), pblh_o(ncol), maxmf_o(ncol), kpbl_c(ncol),       &
      kpbl_o(ncol), ktop_o(ncol))
  do c = 1, ncol
    pin(:, 1, c) = prof(:, 2, c); pin(:, 2, c) = prof(:, 3, c)
    pin(:, 3, c) = prof(:, 5, c); pin(:, 4, c) = prof(:, 6, c)
    pin(:, 5, c) = prof(:, 7, c); pin(:, 6, c) = prof(:, 8, c)
    pin(:, 7, c) = prof(:, 12, c); pin(:, 8, c) = prof(:, 11, c)
  end do
  sin_ = 0.0
  pblh_c = 0.0
  kpbl_c = 1

  open(newunit=uout, file=trim(out_path), access='stream',                &
      form='unformatted', status='new', action='write',                   &
      convert='little_endian')
  write(uout) magic_in, ncol, nz, nstep, delt
  write(uout) scal, prof

  do s = 1, nstep
    initflag = 0
    if (s == 1) initflag = 1
    write(uout) initflag
    do c = 1, ncol
      dz3(1, :, 1) = prof(:, 1, c)
      u(1, :, 1) = pin(:, 1, c); v(1, :, 1) = pin(:, 2, c)
      w(1, :, 1) = prof(:, 4, c); th(1, :, 1) = pin(:, 3, c)
      qv(1, :, 1) = pin(:, 4, c); qc(1, :, 1) = pin(:, 5, c)
      qi(1, :, 1) = pin(:, 6, c)
      qnc = 0.0; qni = 0.0; qnwfa = 0.0; qnifa = 0.0
      p(1, :, 1) = prof(:, 9, c); exner(1, :, 1) = prof(:, 10, c)
      rho(1, :, 1) = pin(:, 8, c); t3d(1, :, 1) = pin(:, 7, c)
      qke(1, :, 1) = sin_(:, 1, c); qke_adv = 0.0
      tsq(1, :, 1) = sin_(:, 2, c); qsq(1, :, 1) = sin_(:, 3, c)
      cov(1, :, 1) = sin_(:, 4, c); el_pbl(1, :, 1) = sin_(:, 5, c)
      sh3d(1, :, 1) = sin_(:, 6, c)
      qc_bl(1, :, 1) = sin_(:, 7, c); cldfra_bl(1, :, 1) = sin_(:, 8, c)
      rthraten(1, :, 1) = prof(:, 13, c)
      pattern_spp_pbl = 0.0
      rublten = 0.0; rvblten = 0.0; rthblten = 0.0; rqvblten = 0.0
      rqcblten = 0.0; rqiblten = 0.0; rqncblten = 0.0; rqniblten = 0.0
      rqnwfablten = 0.0; rqnifablten = 0.0
      exch_h = 0.0; exch_m = 0.0
      dqke = 0.0; qwt = 0.0; qshear = 0.0; qbuoy = 0.0; qdiss = 0.0
      edmf_a = 0.0; edmf_w = 0.0; edmf_qt = 0.0; edmf_thl = 0.0
      edmf_ent = 0.0; edmf_qc = 0.0
      dx = scal(1, c)
      znt(1, 1) = scal(2, c); xland(1, 1) = scal(3, c)
      ts(1, 1) = scal(4, c); qsfc(1, 1) = scal(5, c); qcg(1, 1) = 0.0
      ps(1, 1) = scal(6, c); ust(1, 1) = scal(7, c); ch(1, 1) = scal(8, c)
      hfx(1, 1) = scal(9, c); qfx(1, 1) = scal(10, c)
      wspd(1, 1) = scal(11, c); rmol(1, 1) = scal(12, c)
      uoce = 0.0; voce = 0.0; vdfg = 0.0
      pblh(1, 1) = pblh_c(c); kpbl(1, 1) = kpbl_c(c)
      wstar = 0.0; delta = 0.0
      nupdraft = 0; ktop_plume = 0; kbot_shallow = 0
      maxmf = 0.0; mf_at_base = 0.0

      call mynn_bl_driver(                                                 &
          initflag, .false., .false.,                                      &
          0,                                                               &
          delt, dz3, dx, znt,                                              &
          u, v, w, th, qv, qc, qi, qnc, qni,                               &
          qnwfa, qnifa,                                                    &
          p, exner, rho, t3d,                                              &
          xland, ts, qsfc, qcg, ps,                                        &
          ust, ch, hfx, qfx, rmol, wspd,                                   &
          uoce, voce,                                                      &
          vdfg,                                                            &
          qke,                                                             &
          qke_adv, .false.,                                                &
          tsq, qsq, cov,                                                   &
          rublten, rvblten, rthblten,                                      &
          rqvblten, rqcblten, rqiblten,                                    &
          rqncblten, rqniblten,                                            &
          rqnwfablten, rqnifablten,                                        &
          exch_h, exch_m,                                                  &
          pblh, kpbl,                                                      &
          el_pbl,                                                          &
          dqke, qwt, qshear, qbuoy, qdiss,                                 &
          wstar, delta,                                                    &
          0,                                                               &
          2, sh3d,                                                         &
          2,                                                               &
          1, qc_bl, cldfra_bl,                                             &
          1,                                                               &
          1, 0,                                                            &
          0,                                                               &
          1, 0,                                                            &
          edmf_a, edmf_w, edmf_qt,                                         &
          edmf_thl, edmf_ent, edmf_qc,                                     &
          nupdraft, maxmf, ktop_plume,                                     &
          kbot_shallow, mf_at_base,                                        &
          0, pattern_spp_pbl,                                              &
          rthraten,                                                        &
          .true., .true., .false.,                                         &
          .false., .false., .false.,                                       &
          ids, ide, jds, jde, kds, kde,                                    &
          ims, ime, jms, jme, kms, kme,                                    &
          its, ite, jts, jte, kts, kte)

      outp(:, 1, c) = rublten(1, :, 1); outp(:, 2, c) = rvblten(1, :, 1)
      outp(:, 3, c) = rthblten(1, :, 1); outp(:, 4, c) = rqvblten(1, :, 1)
      outp(:, 5, c) = rqcblten(1, :, 1); outp(:, 6, c) = rqiblten(1, :, 1)
      outp(:, 7, c) = exch_h(1, :, 1); outp(:, 8, c) = exch_m(1, :, 1)
      outp(:, 9, c) = qke(1, :, 1); outp(:, 10, c) = tsq(1, :, 1)
      outp(:, 11, c) = qsq(1, :, 1); outp(:, 12, c) = cov(1, :, 1)
      outp(:, 13, c) = el_pbl(1, :, 1); outp(:, 14, c) = sh3d(1, :, 1)
      outp(:, 15, c) = qc_bl(1, :, 1); outp(:, 16, c) = cldfra_bl(1, :, 1)
      pblh_o(c) = pblh(1, 1); kpbl_o(c) = kpbl(1, 1)
      maxmf_o(c) = maxmf(1, 1); ktop_o(c) = ktop_plume(1, 1)
    end do
    write(uout) outp, pblh_o, kpbl_o, maxmf_o, ktop_o

    ! carry the driver's state and advance every column
    do c = 1, ncol
      sin_(:, :, c) = outp(:, 9:16, c)
      pblh_c(c) = pblh_o(c); kpbl_c(c) = kpbl_o(c)
      do k = 1, nz
        pin(k, 1, c) = pin(k, 1, c) + delt * outp(k, 1, c)
        pin(k, 2, c) = pin(k, 2, c) + delt * outp(k, 2, c)
        pin(k, 3, c) = pin(k, 3, c) + delt * outp(k, 3, c)
        pin(k, 4, c) = max(pin(k, 4, c) + delt * outp(k, 4, c), 0.0)
        pin(k, 5, c) = max(pin(k, 5, c) + delt * outp(k, 5, c), 0.0)
        pin(k, 6, c) = max(pin(k, 6, c) + delt * outp(k, 6, c), 0.0)
        pin(k, 7, c) = pin(k, 3, c) * prof(k, 10, c)
        pin(k, 8, c) = prof(k, 9, c) / (287.0 * pin(k, 7, c))
      end do
    end do
  end do
  close(uout)
end program run_mynn_bl_driver_columns_gsd41_oracle
