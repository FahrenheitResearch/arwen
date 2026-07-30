program run_ruc_transf_oracle
  use module_sf_ruclsm, only: transf, ruclsm_soilvegparm, drysmc, maxsmc, &
      refsmc, wltsmc, pctbl, rstbl, rgltbl, rsmax_data
  implicit none

  integer, parameter :: ncase = 4, nzs = 9
  character(len=16), parameter :: names(ncase) = [character(len=16) :: &
      'wet_forest', 'wilt_dark_grass', 'mixed_hot_crop', 'bare_high_sun']
  integer, parameter :: soil_category(ncase) = [1, 4, 6, 8]
  integer, parameter :: land_category(ncase) = [1, 10, 12, 16]
  integer, parameter :: nroot_case(ncase) = [8, 6, 6, 4]
  real, parameter :: tabs_case(ncase) = [295.0, 280.0, 310.0, 315.0]
  real, parameter :: lai_case(ncase) = [6.2, 0.75, 2.5, 0.75]
  real, parameter :: gswin_case(ncase) = [500.0, 0.0, 40.0, 1200.0]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  character(len=1024) :: output_path
  integer :: n, k, isltyp, iland, unit
  real :: dqm, qmin, ref, wilt, pc, transum
  real :: soiliqw(nzs), zshalf(nzs), tranf(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_transf OUTPUT.csv'
    error stop 2
  end if

  call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,isltyp,iland,nroot,tabs,lai,gswin,dqm,qmin,ref,wilt,pc,rstbl,rgltbl,rsmax,soiliqw,zshalf,tranf,transum'
  do n = 1, ncase
    isltyp = soil_category(n)
    iland = land_category(n)
    dqm = maxsmc(isltyp) - drysmc(isltyp)
    qmin = drysmc(isltyp)
    ref = refsmc(isltyp)
    wilt = wltsmc(isltyp)
    pc = pctbl(iland)
    select case (n)
    case (1)
      soiliqw = 0.25
    case (2)
      soiliqw = 0.02
    case (3)
      soiliqw = [0.05, 0.12, 0.22, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
    case (4)
      soiliqw = [0.01, 0.03, 0.08, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
    end select
    call transf(1, n, nzs, nroot_case(n), soiliqw, tabs_case(n), lai_case(n), &
        gswin_case(n), dqm, qmin, ref, wilt, zshalf, pc, iland, &
        tranf, transum)
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, isltyp, iland, &
          nroot_case(n), &
          tabs_case(n), lai_case(n), gswin_case(n), dqm, qmin, ref, &
          wilt, pc, rstbl(iland), rgltbl(iland), rsmax_data, soiliqw(k), &
          zshalf(k), tranf(k), transum
    end do
  end do
  close(unit)
end program run_ruc_transf_oracle
