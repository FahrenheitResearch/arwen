program run_ruc_soilmoist_oracle
  use module_sf_ruclsm, only: soilmoist, ruclsm_soilvegparm, drysmc, &
      maxsmc, refsmc, satdk
  implicit none

  integer, parameter :: ncase = 4, nzs = 9, nddzs = 14
  character(len=16), parameter :: names(ncase) = [character(len=16) :: &
      'rain_wet', 'dry_evap', 'dew', 'frozen_melt']
  integer, parameter :: soil_category(ncase) = [1, 4, 6, 8]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, isltyp, unit
  real :: delt, x, dqm, qmin, ref, ksat
  real :: qsg, qvg, qcg, qcatm, qvatm, prcp, qkms, drip, dew, smelt
  real :: vegfrac, snowfrac, soilres, ras, riw
  real :: mavail, runoff, runoff2, infiltrp, infmax
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs)
  real :: diffu(nzs), hydro(nzs), transp(nzs), soilice(nzs)
  real :: soilmois(nzs), soiliqw(nzs), soilmois_before(nzs)
  real :: soiliqw_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_soilmoist OUTPUT.csv'
    error stop 2
  end if

  call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
  delt = 60.0
  riw = 0.9
  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do
  dtdzs = 0.0
  dtdzs2 = 0.0
  do k = 2, nzs - 1
    k1 = 2 * k - 3
    k2 = k1 + 1
    x = delt / 2.0 / (zshalf(k + 1) - zshalf(k))
    dtdzs(k1) = x / (zsmain(k) - zsmain(k - 1))
    dtdzs2(k - 1) = x
    dtdzs(k2) = x / (zsmain(k + 1) - zsmain(k))
  end do

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,isltyp,delt,dqm,qmin,ref,ksat,qsg,qvg,qcg,qcatm,qvatm,prcp,qkms,drip,dew,smelt,vegfrac,snowfrac,soilres,ras,riw,zsmain,zshalf,diffu,hydro,transp,soilice,soilmois_before,soiliqw_before,soilmois_after,soiliqw_after,mavail,runoff,runoff2,infiltrp,infmax'
  do n = 1, ncase
    isltyp = soil_category(n)
    dqm = maxsmc(isltyp) - drysmc(isltyp)
    qmin = drysmc(isltyp)
    ref = refsmc(isltyp)
    ksat = satdk(isltyp)
    drip = 0.0
    dew = 0.0
    smelt = 0.0
    soilres = 1.0
    ras = 0.0012
    transp = 0.0
    soilice = 0.0
    select case (n)
    case (1)
      qsg = 0.012
      qvg = 0.010
      qcg = 0.0
      qvatm = 0.009
      qcatm = 0.0
      prcp = -2.0e-5
      qkms = 0.01
      vegfrac = 0.70
      snowfrac = 0.0
      do k = 1, nzs
        soilmois(k) = 0.32 - 0.005 * real(k - 1)
        soiliqw(k) = soilmois(k)
        diffu(k) = 3.0e-5 * (1.0 - 0.05 * real(k - 1))
        hydro(k) = 1.0e-5 * (1.0 - 0.08 * real(k - 1))
      end do
    case (2)
      qsg = 0.015
      qvg = 0.010
      qcg = 0.0
      qvatm = 0.005
      qcatm = 0.0
      prcp = 0.0
      qkms = 0.02
      vegfrac = 0.50
      snowfrac = 0.0
      do k = 1, nzs
        soilmois(k) = 0.06 + 0.002 * real(k - 1)
        soiliqw(k) = soilmois(k)
        diffu(k) = 1.0e-8
        hydro(k) = 0.0
        if (k <= 4) transp(k) = -1.0e-8
      end do
    case (3)
      qsg = 0.008
      qvg = 0.006
      qcg = 0.001
      qvatm = 0.012
      qcatm = 0.001
      prcp = 0.0
      qkms = 0.01
      dew = 2.0e-5
      vegfrac = 0.40
      snowfrac = 0.0
      do k = 1, nzs
        soilmois(k) = 0.20 + 0.003 * real(k - 1)
        soiliqw(k) = soilmois(k)
        diffu(k) = 1.0e-6
        hydro(k) = 1.0e-7
      end do
    case (4)
      qsg = 0.004
      qvg = 0.003
      qcg = 0.0
      qvatm = 0.003
      qcatm = 0.0
      prcp = 0.0
      qkms = 0.005
      smelt = -1.0e-5
      vegfrac = 0.20
      snowfrac = 1.0
      do k = 1, nzs
        soilmois(k) = 0.38
        soiliqw(k) = 0.005
        soilice(k) = 0.375 / riw
        diffu(k) = 0.0
        hydro(k) = 0.0
      end do
    end select
    soilmois_before = soilmois
    soiliqw_before = soiliqw
    mavail = -999.0
    runoff = -999.0
    runoff2 = -999.0
    infiltrp = -999.0
    infmax = -999.0
    call soilmoist(delt, nzs, nddzs, dtdzs, dtdzs2, riw, zsmain, &
        zshalf, diffu, hydro, qsg, qvg, qcg, qcatm, qvatm, prcp, &
        qkms, transp, drip, dew, smelt, soilice, vegfrac, snowfrac, &
        soilres, dqm, qmin, ref, ksat, ras, infmax, soilmois, soiliqw, &
        mavail, runoff, runoff2, infiltrp)
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, isltyp, delt, dqm, &
          qmin, ref, ksat, qsg, qvg, qcg, qcatm, qvatm, prcp, qkms, &
          drip, dew, smelt, vegfrac, snowfrac, soilres, ras, riw, &
          zsmain(k), zshalf(k), diffu(k), hydro(k), transp(k), &
          soilice(k), soilmois_before(k), soiliqw_before(k), &
          soilmois(k), soiliqw(k), mavail, runoff, runoff2, infiltrp, &
          infmax
    end do
  end do
  close(unit)
end program run_ruc_soilmoist_oracle
