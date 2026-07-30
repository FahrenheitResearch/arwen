program run_ruc_soilprop_oracle
  use module_sf_ruclsm, only: soilprop, ruclsm_soilvegparm, bb, drysmc, &
      hc, maxsmc, refsmc, satpsi, satdk, qtz
  implicit none

  integer, parameter :: ncase = 4, nzs = 9
  character(len=16), parameter :: names(ncase) = [character(len=16) :: &
      'warm_wet', 'warm_dry', 'frozen', 'deep_frozen_keep']
  integer, parameter :: soil_category(ncase) = [1, 4, 6, 8]
  character(len=1024) :: output_path
  integer :: n, k, category, unit
  real :: qwrtz, rhocs, dqm, qmin, psis, bclh, ksat, ws
  real :: fwsat(nzs), lwsat(nzs), tav(nzs), keepfr(nzs)
  real :: soilmois(nzs), soiliqw(nzs), soilice(nzs)
  real :: soilmoism(nzs), soiliqwm(nzs), soilicem(nzs)
  real :: thdif(nzs), diffu(nzs), hydro(nzs), cap(nzs)
  real :: rstochcol(nzs), fieldcol_sf(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_soilprop OUTPUT.csv'
    error stop 2
  end if

  call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,isltyp,qwrtz,rhocs,dqm,qmin,psis,bclh,ksat,fwsat,lwsat,tav,keepfr,soilmois,soiliqw,soilice,soilmoism,soiliqwm,soilicem,thdif,diffu,hydro,cap'

  do n = 1, ncase
    category = soil_category(n)
    qwrtz = qtz(category)
    rhocs = hc(category) * 1.e6
    dqm = maxsmc(category) - drysmc(category)
    qmin = drysmc(category)
    psis = -satpsi(category)
    bclh = bb(category)
    ksat = satdk(category)
    ws = dqm + qmin
    do k = 1, nzs
      select case (n)
      case (1)
        tav(k) = 285.0 - 0.5 * real(k - 1)
        soilmois(k) = 0.30 - 0.005 * real(k - 1)
        soiliqw(k) = soilmois(k)
        soilice(k) = 0.0
        soilmoism(k) = 0.2975 - 0.005 * real(k - 1)
        soiliqwm(k) = soilmoism(k)
        soilicem(k) = 0.0
        fwsat(k) = 0.0
        lwsat(k) = ws
        keepfr(k) = 0.0
      case (2)
        tav(k) = 300.0 - 0.25 * real(k - 1)
        soilmois(k) = 0.055 + 0.002 * real(k - 1)
        soiliqw(k) = soilmois(k)
        soilice(k) = 0.0
        soilmoism(k) = 0.056 + 0.002 * real(k - 1)
        soiliqwm(k) = soilmoism(k)
        soilicem(k) = 0.0
        fwsat(k) = 0.0
        lwsat(k) = ws
        keepfr(k) = 0.0
      case (3)
        tav(k) = 269.0 + 0.3 * real(k - 1)
        soilmois(k) = 0.32
        soiliqw(k) = 0.16
        soilice(k) = 0.16 / 0.9
        soilmoism(k) = 0.31
        soiliqwm(k) = 0.155
        soilicem(k) = 0.155 / 0.9
        fwsat(k) = dqm - soiliqwm(k)
        lwsat(k) = soiliqwm(k) + qmin
        keepfr(k) = 0.0
      case (4)
        tav(k) = 267.0 + 0.4 * real(k - 1)
        soilmois(k) = 0.38
        soiliqw(k) = 0.005
        soilice(k) = 0.375 / 0.9
        soilmoism(k) = 0.377
        soiliqwm(k) = 0.005
        soilicem(k) = 0.372 / 0.9
        fwsat(k) = dqm - soiliqwm(k)
        lwsat(k) = soiliqwm(k) + qmin
        keepfr(k) = 1.0
      end select
    end do
    thdif = 0.0
    diffu = 0.0
    hydro = 0.0
    cap = 0.0
    rstochcol = 0.0
    fieldcol_sf = 0.0
    call soilprop(0, rstochcol, fieldcol_sf, nzs, fwsat, lwsat, tav, &
        keepfr, soilmois, soiliqw, soilice, soilmoism, soiliqwm, &
        soilicem, qwrtz, rhocs, dqm, qmin, psis, bclh, ksat, 0.9, &
        3.35e5, 1004.5, 9.81, 4.183e6, 1.89e6, 7.7, 2.2, 0.57, &
        thdif, diffu, hydro, cap)
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, category, qwrtz, &
          rhocs, dqm, qmin, psis, bclh, ksat, fwsat(k), lwsat(k), &
          tav(k), keepfr(k), soilmois(k), soiliqw(k), soilice(k), &
          soilmoism(k), soiliqwm(k), soilicem(k), thdif(k), diffu(k), &
          hydro(k), cap(k)
    end do
  end do
  close(unit)
end program run_ruc_soilprop_oracle
