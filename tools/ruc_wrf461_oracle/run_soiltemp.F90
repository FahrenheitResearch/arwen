program run_ruc_soiltemp_oracle
  use module_sf_ruclsm, only: soiltemp
  implicit none

  integer, parameter :: ncase = 4, nzs = 9, nddzs = 14
  character(len=20), parameter :: names(ncase) = [character(len=20) :: &
      'moist_saturated', 'dry_second_solve', 'warm_rain', &
      'humid_condensing']
  integer, parameter :: nroot_case(ncase) = [8, 6, 4, 8]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  character(len=1024) :: output_path, table_path
  integer :: n, k, k1, k2, unit, table_unit
  real :: delt, xgeom, cq, evs, eis, r61
  real :: conflx, prcpms, rainf, patm, tabs, qvatm, qcatm
  real :: emiss, rnet, qkms, tkms, pc, rho, vegfrac, lai
  real :: drycan, wetcan, transum, dew, mavail, soilres, alfa
  real :: dqm, qmin, bclh, soilt, qvg, qsg, qcg, storage
  real :: soilt_before, qvg_before
  real :: zshalf(nzs), dtdzs(nddzs), tbq(5001)
  real :: thdif(nzs), cap(nzs), tso(nzs), tso_before(nzs)
  real, parameter :: xlv = 2.5e6, cp = 1004.5, g0_p = 9.81
  real, parameter :: cvw = 4183.0, stbolt = 5.67051e-8

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_soiltemp OUTPUT.csv'
    error stop 2
  end if
  call get_command_argument(2, table_path)

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
  if (len_trim(table_path) > 0) then
    open(newunit=table_unit, file=trim(table_path), status='replace', &
        action='write')
    write(table_unit, '(A)') 'k,tbq'
    do k = 1, 5001
      write(table_unit, '(*(g0,:,","))') k, tbq(k)
    end do
    close(table_unit)
  end if

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,conflx,nroot,prcpms,rainf,patm,tabs,qvatm,qcatm,emiss,rnet,qkms,tkms,pc,rho,vegfrac,lai,drycan,wetcan,transum,dew,mavail,soilres,alfa,dqm,qmin,bclh,thdif,cap,tso_before,tso_after,soilt_before,soilt_after,qvg_before,qvg_after,qsg_after,qcg_after,storage'

  do n = 1, ncase
    conflx = 0.5
    prcpms = 0.0
    rainf = 0.0
    patm = 0.95
    tabs = 294.0
    qvatm = 0.010
    qcatm = 0.0
    emiss = 0.95
    rnet = 180.0
    qkms = 0.012
    tkms = 0.010
    pc = 0.7
    rho = 1.10
    vegfrac = 0.65
    lai = 2.5
    drycan = 0.80
    wetcan = 0.05
    transum = 0.04
    dew = 0.0
    mavail = 0.70
    soilres = 1.0
    alfa = 1.0
    dqm = 0.40
    qmin = 0.02
    bclh = 4.5
    soilt = 293.0
    qvg = 0.011
    qsg = 0.015
    qcg = 0.0
    do k = 1, nzs
      tso(k) = 293.0 - 0.75 * real(k - 1)
      thdif(k) = 8.0e-7 * (1.0 - 0.04 * real(k - 1))
      cap(k) = 2.10e6 + 2.5e4 * real(k - 1)
    end do

    select case (n)
    case (1)
      mavail = 1.0
      qvatm = 0.012
      rnet = 220.0
    case (2)
      tabs = 303.0
      qvatm = 0.003
      mavail = 0.12
      vegfrac = 0.25
      drycan = 0.95
      wetcan = 0.0
      transum = 0.01
      rnet = 500.0
      soilt = 301.0
      qvg = 0.006
      do k = 1, nzs
        tso(k) = 301.0 - 0.60 * real(k - 1)
      end do
    case (3)
      prcpms = 2.5e-5
      rainf = 1.0
      tabs = 289.0
      qvatm = 0.009
      mavail = 0.78
      rnet = 90.0
      soilt = 291.0
      qvg = 0.010
      do k = 1, nzs
        tso(k) = 291.0 - 0.45 * real(k - 1)
      end do
    case (4)
      tabs = 298.0
      qvatm = 0.026
      mavail = 0.90
      rnet = 330.0
      soilt = 297.0
      qvg = 0.019
      do k = 1, nzs
        tso(k) = 297.0 - 0.80 * real(k - 1)
      end do
    end select

    tso_before = tso
    soilt_before = soilt
    qvg_before = qvg
    call soiltemp(1, 1, 1, 1, delt, 1, conflx, nzs, nddzs, nroot_case(n), &
        prcpms, rainf, patm, tabs, qvatm, qcatm, emiss, rnet, qkms, &
        tkms, pc, rho, vegfrac, lai, thdif, cap, drycan, wetcan, &
        transum, dew, mavail, soilres, alfa, dqm, qmin, bclh, zsmain, &
        zshalf, dtdzs, tbq, xlv, cp, g0_p, cvw, stbolt, tso, soilt, &
        qvg, qsg, qcg, storage)
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, conflx, &
          nroot_case(n), prcpms, rainf, patm, tabs, qvatm, qcatm, emiss, rnet, &
          qkms, tkms, pc, rho, vegfrac, lai, drycan, wetcan, transum, &
          dew, mavail, soilres, alfa, dqm, qmin, bclh, thdif(k), cap(k), &
          tso_before(k), tso(k), soilt_before, soilt, qvg_before, qvg, &
          qsg, qcg, storage
    end do
  end do
  close(unit)
end program run_ruc_soiltemp_oracle
