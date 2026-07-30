program run_ruc_soilvegin_oracle
  use module_sf_ruclsm, only: ruclsm_soilvegparm, soilvegin
  implicit none

  integer, parameter :: ncase = 6, nlcat = 21, nscat = 19
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'evergreen_cold', 'evergreen_warm', 'crop_midseason', &
      'water_preserve_znt', 'lai2d_preserve', 'grass_short_season']
  character(len=1024) :: output_path
  integer :: n, unit
  integer :: isltyp(ncase), ivgtyp(ncase), iforest
  real :: shdmin(ncase), shdmax(ncase), vegfrac(ncase)
  real :: znt_before(ncase), lai_before(ncase)
  logical :: rdlai2d(ncase)
  real :: lufrac(nlcat), soilfrac(nscat)
  real :: emiss, pc, znt, lai
  real :: qwrtz, rhocs, bclh, dqm, ksat, psis, qmin, ref, wilt

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_soilvegin OUTPUT.csv'
    error stop 2
  end if

  call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')

  isltyp = [1, 4, 6, 14, 19, 16]
  ivgtyp = [1, 1, 12, 17, 14, 10]
  shdmin = [10.0, 10.0, 15.0, 0.0, 20.0, 20.0]
  shdmax = [80.0, 80.0, 75.0, 100.0, 90.0, 20.5]
  vegfrac = [10.0, 80.0, 45.0, 0.0, 55.0, 20.25]
  znt_before = [0.333, 0.444, 0.555, 0.00037, 0.777, 0.888]
  lai_before = [9.1, 9.2, 9.3, 9.4, 2.345, 9.6]
  rdlai2d = [.false., .false., .false., .false., .true., .false.]

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,isltyp,ivgtyp,shdmin,shdmax,vegfrac,znt_before,lai_before,rdlai2d,iforest,emiss,pc,znt,lai,qwrtz,rhocs,bclh,dqm,ksat,psis,qmin,ref,wilt'
  do n = 1, ncase
    lufrac = 0.0
    soilfrac = 0.0
    lufrac(ivgtyp(n)) = 1.0
    soilfrac(isltyp(n)) = 1.0
    znt = znt_before(n)
    lai = lai_before(n)
    emiss = -999.0
    pc = -999.0
    call soilvegin(0, 0, soilfrac, nscat, shdmin(n), shdmax(n), &
        nlcat, ivgtyp(n), isltyp(n), 17, .true., iforest, lufrac, &
        vegfrac(n), emiss, pc, znt, lai, rdlai2d(n), qwrtz, rhocs, &
        bclh, dqm, ksat, psis, qmin, ref, wilt, n, 1)
    write(unit, '(*(g0,:,","))') trim(names(n)), isltyp(n), ivgtyp(n), &
        shdmin(n), shdmax(n), vegfrac(n), znt_before(n), lai_before(n), &
        merge(1, 0, rdlai2d(n)), iforest, emiss, pc, znt, lai, qwrtz, &
        rhocs, bclh, dqm, ksat, psis, qmin, ref, wilt
  end do
  close(unit)
end program run_ruc_soilvegin_oracle
