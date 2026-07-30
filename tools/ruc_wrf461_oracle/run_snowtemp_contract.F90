program run_ruc_snowtemp_contract_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::snowtemp on the snow
  ! regimes oracle/snowtemp.csv leaves unexercised.  Same harness, same
  ! pinned module, same compiler and the same CSV schema as run_snowtemp.F90;
  ! this file only adds regimes, and oracle/snowtemp.csv is untouched.
  !
  ! A mutation study over the gpuwm port (one read of one argument replaced
  ! per mutant) showed that snowtemp.csv, for all its six regimes, leaves 72
  ! argument read sites and 6 reachable if-arms undetectable.  Every regime
  ! below is chosen to kill a named group of those:
  !
  !   1 melt_dense_dry_evap     - the evaporation half of the melt energy
  !                               budget (:5436-5450, the only reader of
  !                               tranf/transp/ett1/edir1/ec1), with rhosn
  !                               >= 350 so the 22apr22 exemption at :5499
  !                               skips the Egglston limiter.  A limited
  !                               smelt saturates and masks rnet, qfx, hfx,
  !                               soh and x entirely.  snowfrac < 1 so the
  !                               (1-snowfrac) weights at :5416 and :5418
  !                               are not degenerate and the :5373-5377
  !                               full-cover clamps do not fire.
  !   2 melt_trace_all_evap     - :5483-5491, all the remaining snow can
  !                               evaporate: beta is rewritten inside the
  !                               melt block and the jump to :5518 skips the
  !                               limiter and the rr cap.
  !   3 melt_blended_trace      - melt on snow blended into the top soil
  !                               layer: the blended soh at :5459, and a
  !                               pack thin enough that the rr cap at
  !                               :5510-5512 binds and :5533 sends rsm to
  !                               zero.
  !   4 bottom_melt_2layer      - bottom melt under a two-layer pack, so
  !                               hsn = snhei-deltsn at :5638, with rhosn
  !                               >= 350 so the 5.8e-9 bottom limit at
  !                               :5659 does not mask cap(1), zshalf(2),
  !                               rhocsn or hsn, and snowfrac < 1 so the
  !                               :5649 weighting is not degenerate.
  !   5 bottom_melt_trace       - top melt under partial cover, so the
  !                               :5373-5377 clamps do not fire and tso(1)
  !                               stays above 273.15, followed by bottom
  !                               melt where rr = snwe/delt at :5663-5664 is
  !                               what limits smeltg rather than the 5.8e-9
  !                               cap.  The pack is gone afterwards, so
  !                               :5677 leaves tso(1) alone.
  !   6 lowdens_new_on_aged     - rhosn >= 156 with fresh low-density new
  !                               snow, so the second disjunct of the
  !                               :5049 keff test is what selects the
  !                               low-density conductivity.
  !   7 melt_dense_moist        - the condensation half of the melt budget
  !                               (:5427-5433) on a dense pack, so the
  !                               Egglston limiter is skipped and snoh
  !                               actually depends on qfx = -xlvm*rho*dew.
  !                               Every other condensation-half melt regime
  !                               has smelt pinned by a limiter, which makes
  !                               snoh = smelt*xlmelt*1.e3 independent of
  !                               qfx and masks xlvm, rho and dew.
  !   8 melt_hot_surface_1layer - the regime the 22apr22 comments at :5497
  !                               and :5657 were written for: rhosn < 350
  !                               but soilt above 283 K, so neither
  !                               Egglston guard applies.  That is the only
  !                               way to get an unlimited melt together
  !                               with a Koren density update (:5583, needs
  !                               rhosn < 350) and an unlimited bottom melt
  !                               with hsn = snhei, which pins the
  !                               post-update rhocsn and thdifsn, the
  !                               :5595 rhonewsn disjunct on the second
  !                               conductivity call, and cap(1)/zshalf(2)
  !                               in snohg.
  !
  ! snowtemp declares ilnb as intent(out) yet reads it at the tsnav update
  ! when the snow layer is blended into the top soil layer, so ilnb is
  ! seeded and ilnb_before recorded, exactly as run_snowtemp.F90 does.
  use module_sf_ruclsm, only: snowtemp
  implicit none

  integer, parameter :: ncase = 8, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'melt_dense_dry_evap', 'melt_trace_all_evap', 'melt_blended_trace', &
      'bottom_melt_2layer', 'bottom_melt_trace', 'lowdens_new_on_aged', &
      'melt_dense_moist', 'melt_hot_surface_1layer']
  integer, parameter :: nroot_case(ncase) = [6, 4, 6, 4, 6, 5, 6, 6]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: g0_p = 9.81, cw = 4.183e6, stbolt = 5.67051e-8
  real, parameter :: xlmelt = 3.35e5
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, ilnb, ilnb_before
  real :: delt, xgeom, cq, evs, eis, r61, conflx
  real :: snwe, snwepr, snhei, newsnow, snowfrac, beta, deltsn, snth
  real :: rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, tabs
  real :: qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, pc, rho
  real :: vegfrac, drycan, wetcan, cst, transum, dew, mavail
  real :: dqm, qmin, psis, bclh, xlvm, rovcp, cvw
  real :: snweprint, snheiprint, rsm, soilt, soilt1, tsnav
  real :: qvg, qsg, qcg, smelt, snoh, snflx, s, x
  real :: umveg, epot, epdt, ras
  real :: snwe_before, snhei_before, beta_before, rhosn_before
  real :: soilt_before, soilt1_before, tsnav_before, dew_before
  real :: qvg_before, qsg_before, qcg_before
  real :: zshalf(nzs), dtdzs(nddzs), tbq(5001)
  real :: thdif(nzs), cap(nzs), tranf(nzs), tso(nzs), tso_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_snowtemp_contract OUTPUT.csv'
    error stop 2
  end if

  delt = 60.0
  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do
  dtdzs = 0.0
  do k = 2, nzs - 1
    k1 = 2 * k - 3
    k2 = k1 + 1
    xgeom = delt / 2.0 / (zshalf(k + 1) - zshalf(k))
    dtdzs(k1) = xgeom / (zsmain(k) - zsmain(k - 1))
    dtdzs(k2) = xgeom / (zsmain(k + 1) - zsmain(k))
  end do

  cq = 173.15 - 0.05
  r61 = 6.1153 * 0.62198
  do k = 1, 5001
    cq = cq + 0.05
    evs = exp(17.67 * (cq - 273.15) / (cq - 29.65))
    eis = exp(22.514 - 6.15e3 / cq)
    if (cq >= 273.15) then
      tbq(k) = r61 * evs
    else
      tbq(k) = r61 * eis
    end if
  end do

  xlvm = xlv + xlmelt
  rovcp = r_d / cp
  cvw = cw
  ktau = 1
  iland = 1
  isoil = 4

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,ktau,conflx,nroot,iland,isoil,snwe_before,snwepr,snhei_before,newsnow,snowfrac,beta_before,deltsn,snth,rhosn_before,rhonewsn,meltfactor,prcpms,rainf,patm,tabs,qvatm,qcatm,glw,gsw,emiss,rnet,qkms,tkms,pc,rho,vegfrac,drycan,wetcan,cst,transum,mavail,dqm,qmin,psis,bclh,xlvm,cp,rovcp,g0_p,cvw,stbolt,zsmain,zshalf,thdif,cap,tranf,tso_before,tso_after,soilt_before,soilt_after,soilt1_before,soilt1_after,tsnav_before,tsnav_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,dew_before,dew_after,snwe_after,snhei_after,rhosn_after,beta_after,smelt,snoh,snflx,s,rsm,snweprint,snheiprint,x,ilnb_before,ilnb_after'

  do n = 1, ncase
    conflx = 40.0
    prcpms = 0.0
    rainf = 0.0
    patm = 0.95
    qcatm = 0.0
    pc = 0.7
    rho = 1.25
    vegfrac = 0.20
    wetcan = 0.02
    drycan = 1.0 - wetcan
    cst = 0.0
    mavail = 0.60
    dqm = 0.40
    qmin = 0.02
    psis = -0.35
    bclh = 5.4
    emiss = 0.98
    meltfactor = 1.0
    newsnow = 0.0
    rhonewsn = 100.0
    snowfrac = 1.0
    ilnb = 1
    dew = 0.0
    smelt = 0.0
    snoh = 0.0
    snflx = 0.0
    s = 0.0
    x = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0
    do k = 1, nzs
      thdif(k) = 6.5e-7 * (1.0 + 0.03 * real(k - 1))
      cap(k) = 1.90e6 + 3.0e4 * real(k - 1)
      tranf(k) = 0.0
    end do

    select case (n)
    case (1)
      ! Dry air over a dense one-layer pack already within 0.05 K of the
      ! melting point: soilt clears 273.15 with beta = 1, and inside the
      ! melt block qvatm < qsg keeps epot > 0, so q1 = epot*ras > 0 and the
      ! evaporation half runs.  rhosn = 400 with no new snow selects the
      ! dense-snow exemption at :5499, so smelt tracks snoh instead of
      ! saturating at delt/60*5.6e-8*meltfactor.
      rhosn = 400.0
      snwe = 0.05
      snowfrac = 0.70
      tabs = 281.0
      qvatm = 0.0025
      glw = 330.0
      gsw = 750.0
      rnet = 200.0
      qkms = 0.006
      tkms = 0.006
      vegfrac = 0.30
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0039997
      qsg = 0.0039997
      do k = 1, nzs
        tso(k) = 272.20 + 0.08 * real(k - 1)
      end do
    case (2)
      ! A trace pack with air only 2.0e-5 kg/kg drier than the surface, so
      ! epot is positive but tiny.  snwepr sits inside (umveg, 1] times
      ! epot*ras*delt: above the harness threshold that would reduce beta
      ! before the call, at or below the :5483 threshold that reduces it
      ! inside the melt block and jumps to :5518.
      rhosn = 150.0
      snwe = 7.6e-9
      tabs = 277.0
      qvatm = 0.00398
      glw = 300.0
      gsw = 260.0
      rnet = 120.0
      qkms = 0.006
      tkms = 0.020
      vegfrac = 0.30
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0040
      qsg = 0.0040
      do k = 1, nzs
        tso(k) = 273.05 + 0.02 * real(k - 1)
      end do
    case (3)
      ! Moist air over a trace pack blended into the top soil layer.  The
      ! condensation half runs, so epot < 0 and the :5483 shortcut is not
      ! taken; the pack is small enough that the rr cap at :5510 is what
      ! limits smelt, and snhei < 0.01 sends rsm to zero at :5533.
      rhosn = 200.0
      snwe = 5.0e-7
      tabs = 278.0
      qvatm = 0.0050
      glw = 320.0
      gsw = 240.0
      rnet = 120.0
      qkms = 0.020
      tkms = 0.020
      vegfrac = 0.25
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0040
      qsg = 0.0040
      do k = 1, nzs
        tso(k) = 273.05 + 0.02 * real(k - 1)
      end do
    case (4)
      ! Cold air over a dense two-layer pack on thawed ground: no top melt,
      ! but tso(1) > 273.15 drives bottom melt with hsn = snhei-deltsn, and
      ! rhosn >= 350 with no new snow skips the 5.8e-9 limit at :5659 so the
      ! bottom heat capacity is actually pinned.  snowfrac < 1 keeps the
      ! :5649 weighting non-degenerate.
      rhosn = 400.0
      snwe = 0.08
      snowfrac = 0.60
      tabs = 265.0
      qvatm = 0.0022
      glw = 220.0
      gsw = 40.0
      rnet = -30.0
      qkms = 0.008
      tkms = 0.007
      vegfrac = 0.05
      soilt = 268.0
      soilt1 = 271.0
      qvg = 0.0025944
      qsg = 0.0025944
      do k = 1, nzs
        tso(k) = 278.0 + 0.10 * real(k - 1)
      end do
    case (5)
      ! A partially covered trace pack under warm, very moist air.  soilt
      ! clears 273.15 so top melt runs, but snowfrac < 1 means the
      ! :5373-5377 clamps do not pull tso(1) back to 273.15, so the bottom
      ! melt gate at :5636 opens on what the top melt left behind.  What is
      ! left is under 3.5e-7 m, so rr = snwe/delt at :5663 is below the
      ! 5.8e-9 empirical cap and rr is what limits smeltg.  That is the only
      ! way in: whenever rhosn < 350 the cap holds smeltg at 5.8e-9, so rr
      ! binds only for a pack this thin, and a pack this thin has fso within
      ! 1e-4 of one, which ties tso(1) to soilt.  Condensation (qvatm > qsg)
      ! nearly cancels the melt loss so the pack survives the first pass.
      rhosn = 200.0
      snwe = 2.6e-7
      snowfrac = 0.50
      tabs = 277.0
      qvatm = 0.0062
      glw = 320.0
      gsw = 150.0
      rnet = 60.0
      qkms = 0.020
      tkms = 0.020
      vegfrac = 0.25
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0040
      qsg = 0.0040
      do k = 1, nzs
        tso(k) = 273.05 + 0.02 * real(k - 1)
      end do
    case (6)
      ! An aged pack (rhosn >= 156) taking fresh low-density snow: the
      ! :5049 conductivity test selects the low-density branch through its
      ! second disjunct, which no other regime does.
      rhosn = 200.0
      snwe = 0.03
      newsnow = 0.003
      rhonewsn = 120.0
      tabs = 263.0
      qvatm = 0.0014
      glw = 200.0
      gsw = 35.0
      rnet = -25.0
      qkms = 0.007
      tkms = 0.006
      vegfrac = 0.15
      soilt = 262.0
      soilt1 = 262.0
      qvg = 0.0015340
      qsg = 0.0015340
      do k = 1, nzs
        tso(k) = 261.0 + 0.70 * real(k - 1)
      end do
    case (7)
      ! Warm moist air over a dense one-layer pack at the melting point.
      ! qvatm > qsg so the condensation half runs, and rhosn >= 350 with no
      ! new snow skips the Egglston limiter, so smelt follows snoh and snoh
      ! follows qfx = -xlvm*rho*dew.
      rhosn = 400.0
      snwe = 0.05
      snowfrac = 0.70
      tabs = 281.0
      qvatm = 0.0060
      glw = 330.0
      gsw = 750.0
      rnet = 200.0
      qkms = 0.006
      tkms = 0.006
      vegfrac = 0.30
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0039997
      qsg = 0.0039997
      do k = 1, nzs
        tso(k) = 272.20 + 0.08 * real(k - 1)
      end do
    case (8)
      ! Strong spring melt-out: full sun and warm moist advection drive the
      ! skin above 283 K over a 0.1 m pack of rhosn = 300 on thawed ground.
      ! Both Egglston guards test `soilt < 283.`, so neither the top nor the
      ! bottom melt is limited, while rhosn < 350 still lets the Koren
      ! liquid retention and density update at :5529-5610 run.  snowfrac is
      ! low enough that soiltfrac, which the second iteration relaxes the
      ! skin onto, stays above 283 K.
      rhosn = 300.0
      snwe = 0.03
      newsnow = 0.002
      rhonewsn = 120.0
      snowfrac = 0.20
      tabs = 305.0
      qvatm = 0.030
      glw = 430.0
      gsw = 1000.0
      rnet = 900.0
      qkms = 0.050
      tkms = 0.050
      soilt = 285.0
      soilt1 = 285.0
      qvg = 0.0075142
      qsg = 0.0075142
      do k = 1, nzs
        tso(k) = 284.0 + 0.20 * real(k - 1)
      end do
    end select

    do k = 1, nroot_case(n)
      tranf(k) = 0.02 * (1.0 - 0.1 * real(k - 1))
    end do
    transum = 0.0
    do k = 1, nroot_case(n)
      transum = transum + tranf(k)
    end do

    snhei = snwe * 1.0e3 / rhosn

    ! --- snowsoil's own pre-call arithmetic, reproduced verbatim ---
    deltsn = 0.05 * 1.e3 / rhosn
    snth = 0.01 * 1.e3 / rhosn
    if (snhei >= deltsn + snth) then
      if (snhei - deltsn - snth < snth) deltsn = 0.5 * (snhei - snth)
    end if
    ras = rho * 1.e-3
    umveg = 1.0 - vegfrac
    snwepr = snwe
    beta = 1.0
    epot = -qkms * (qvatm - qsg)
    epdt = epot * ras * delt * umveg
    if (epdt > 0. .and. snwepr <= epdt) then
      beta = snwepr / max(1.e-8, epdt)
      snwe = 0.0
    end if
    ! --------------------------------------------------------------

    snwe_before = snwe
    snhei_before = snhei
    beta_before = beta
    rhosn_before = rhosn
    tso_before = tso
    soilt_before = soilt
    soilt1_before = soilt1
    tsnav = soilt - 273.15
    tsnav_before = tsnav
    dew_before = dew
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    ilnb_before = ilnb

    call snowtemp(1, n, iland, isoil, delt, ktau, conflx, nzs, nddzs, &
        nroot_case(n), snwe, snwepr, snhei, newsnow, snowfrac, beta, &
        deltsn, snth, rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, &
        tabs, qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, pc, rho, &
        vegfrac, thdif, cap, drycan, wetcan, cst, tranf, transum, dew, &
        mavail, dqm, qmin, psis, bclh, zsmain, zshalf, dtdzs, tbq, xlvm, &
        cp, rovcp, g0_p, cvw, stbolt, snweprint, snheiprint, rsm, tso, &
        soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, s, &
        ilnb, x)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          nroot_case(n), iland, isoil, snwe_before, snwepr, snhei_before, &
          newsnow, snowfrac, beta_before, deltsn, snth, rhosn_before, &
          rhonewsn, meltfactor, prcpms, rainf, patm, tabs, qvatm, qcatm, &
          glw, gsw, emiss, rnet, qkms, tkms, pc, rho, vegfrac, drycan, &
          wetcan, cst, transum, mavail, dqm, qmin, psis, bclh, xlvm, cp, &
          rovcp, g0_p, cvw, stbolt, zsmain(k), zshalf(k), thdif(k), &
          cap(k), tranf(k), tso_before(k), tso(k), soilt_before, soilt, &
          soilt1_before, soilt1, tsnav_before, tsnav, qvg_before, qvg, &
          qsg_before, qsg, qcg_before, qcg, dew_before, dew, snwe, &
          snhei, rhosn, beta, smelt, snoh, snflx, s, rsm, snweprint, &
          snheiprint, x, ilnb_before, ilnb
    end do
  end do
  close(unit)
end program run_ruc_snowtemp_contract_oracle
