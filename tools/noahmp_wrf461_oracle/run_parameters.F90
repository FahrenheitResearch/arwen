! Pin WRF v4.6.1 NOAHMP_TABLES + TRANSFER_MP_PARAMETERS against byte-pinned tables.
!
! Routines exercised, all from unmodified pinned sources:
!   NOAHMP_TABLES::read_mp_veg_parameters      (phys/module_sf_noahmplsm.F)
!   NOAHMP_TABLES::read_mp_soil_parameters
!   NOAHMP_TABLES::read_mp_rad_parameters
!   NOAHMP_TABLES::read_mp_global_parameters
!   NOAHMP_TABLES::read_mp_crop_parameters
!   NOAHMP_TABLES::read_tiledrain_parameters
!   NOAHMP_TABLES::read_mp_optional_parameters
!   module_sf_noahmpdrv::TRANSFER_MP_PARAMETERS (phys/module_sf_noahmpdrv.F)
!
! The reader call order is exactly module_sf_noahmpdrv.F NOAHMP_INIT lines
! 2000-2007 under the pinned option identity, which has iopt_irr = 0 and
! therefore never calls read_mp_irrigation_parameters.
!
! This slice does NOT admit:
!   * the irrigation parameter block: with iopt_irr = 0 the IRR_*/FILOSS/
!     SPRIR_RATE/MICIR_RATE/FIRTFAC/IR_RAIN table entries are never assigned by
!     WRF, so parameters%IRR_* is formally undefined and is not emitted;
!   * the crop parameter block: CROPTYPE = 0 under opt_crop = 0 skips the crop
!     branch of TRANSFER_MP_PARAMETERS, so parameters%PLTDAY..BIO2LAI is
!     formally undefined and is not emitted;
!   * parameters%SLAREA and parameters%EPS, which MPTABLE.TBL supplies but
!     TRANSFER_MP_PARAMETERS never assigns;
!   * parameters%FRZX for SOILTYPE(1) == 14 (water), which WRF leaves unset;
!   * OPT_SOIL /= 1 (depth-varying, pedotransfer or prescribed soil), and the
!     USGS land-use table, which this harness reads but does not transfer.

program run_noahmp_parameters_oracle
  use module_sf_noahmplsm, only: noahmp_parameters
  use noahmp_tables, only: read_mp_veg_parameters, read_mp_soil_parameters, &
      read_mp_rad_parameters, read_mp_global_parameters, &
      read_mp_crop_parameters, read_tiledrain_parameters, &
      read_mp_optional_parameters
  use module_sf_noahmpdrv, only: transfer_mp_parameters
  implicit none

  integer, parameter :: nsoil = 4
  integer, parameter :: ncase = 7
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'evergreen_needleleaf', 'grassland', 'cropland', 'urban', &
      'barren', 'snow_ice', 'water']
  integer, parameter :: vegtypes(ncase)   = [ 1, 10, 12, 13, 16, 15, 17]
  integer, parameter :: soiltypes(ncase)  = [ 6,  3,  8,  6,  1, 16, 12]
  integer, parameter :: slopetypes(ncase) = [ 1,  2,  1,  1,  9,  1,  4]
  integer, parameter :: soilcolors(ncase) = [ 4,  4,  4,  4,  1,  8,  4]

  character(len=1024) :: output_path
  integer :: n, unit
  integer :: soiltype(nsoil)
  type(noahmp_parameters) :: p

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_parameters OUTPUT.csv'
    error stop 2
  end if

  ! Exactly module_sf_noahmpdrv.F NOAHMP_INIT lines 2000-2007 with iopt_irr = 0.
  call read_mp_veg_parameters('MODIFIED_IGBP_MODIS_NOAH')
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()
  call read_mp_crop_parameters()
  call read_tiledrain_parameters()
  call read_mp_optional_parameters()

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,vegtype,soiltype,slopetype,soilcolor,field,index,value'

  do n = 1, ncase
    soiltype = soiltypes(n)
    call transfer_mp_parameters(nsoil, vegtypes(n), soiltype, slopetypes(n), &
        soilcolors(n), 0, p)

    ! Category identity flags.
    call emit_int(n, 'ISWATER',   0, p%ISWATER)
    call emit_int(n, 'ISBARREN',  0, p%ISBARREN)
    call emit_int(n, 'ISICE',     0, p%ISICE)
    call emit_int(n, 'ISCROP',    0, p%ISCROP)
    call emit_int(n, 'EBLFOREST', 0, p%EBLFOREST)
    call emit_int(n, 'URBAN_FLAG', 0, merge(1, 0, p%URBAN_FLAG))

    ! MPTABLE vegetation block.
    call emit(n, 'CH2OP',  0, p%CH2OP)
    call emit(n, 'DLEAF',  0, p%DLEAF)
    call emit(n, 'Z0MVT',  0, p%Z0MVT)
    call emit(n, 'HVT',    0, p%HVT)
    call emit(n, 'HVB',    0, p%HVB)
    call emit(n, 'DEN',    0, p%DEN)
    call emit(n, 'RC',     0, p%RC)
    call emit(n, 'MFSNO',  0, p%MFSNO)
    call emit(n, 'SCFFAC', 0, p%SCFFAC)
    call emit(n, 'CBIOM',  0, p%CBIOM)
    call emit_array(n, 'SAIM', p%SAIM, 12)
    call emit_array(n, 'LAIM', p%LAIM, 12)
    call emit(n, 'SLA',    0, p%SLA)
    call emit(n, 'DILEFC', 0, p%DILEFC)
    call emit(n, 'DILEFW', 0, p%DILEFW)
    call emit(n, 'FRAGR',  0, p%FRAGR)
    call emit(n, 'LTOVRC', 0, p%LTOVRC)
    call emit(n, 'C3PSN',  0, p%C3PSN)
    call emit(n, 'KC25',   0, p%KC25)
    call emit(n, 'AKC',    0, p%AKC)
    call emit(n, 'KO25',   0, p%KO25)
    call emit(n, 'AKO',    0, p%AKO)
    call emit(n, 'VCMX25', 0, p%VCMX25)
    call emit(n, 'AVCMX',  0, p%AVCMX)
    call emit(n, 'BP',     0, p%BP)
    call emit(n, 'MP',     0, p%MP)
    call emit(n, 'QE25',   0, p%QE25)
    call emit(n, 'AQE',    0, p%AQE)
    call emit(n, 'RMF25',  0, p%RMF25)
    call emit(n, 'RMS25',  0, p%RMS25)
    call emit(n, 'RMR25',  0, p%RMR25)
    call emit(n, 'ARM',    0, p%ARM)
    call emit(n, 'FOLNMX', 0, p%FOLNMX)
    call emit(n, 'TMIN',   0, p%TMIN)
    call emit(n, 'XL',     0, p%XL)
    call emit_array(n, 'RHOL', p%RHOL, 2)
    call emit_array(n, 'RHOS', p%RHOS, 2)
    call emit_array(n, 'TAUL', p%TAUL, 2)
    call emit_array(n, 'TAUS', p%TAUS, 2)
    call emit(n, 'MRP',    0, p%MRP)
    call emit(n, 'CWPVT',  0, p%CWPVT)
    call emit(n, 'WRRAT',  0, p%WRRAT)
    call emit(n, 'WDPOOL', 0, p%WDPOOL)
    call emit(n, 'TDLEF',  0, p%TDLEF)
    call emit_int(n, 'NROOT', 0, p%NROOT)
    call emit(n, 'RGL',    0, p%RGL)
    call emit(n, 'RSMIN',  0, p%RSMIN)
    call emit(n, 'HS',     0, p%HS)
    call emit(n, 'TOPT',   0, p%TOPT)
    call emit(n, 'RSMAX',  0, p%RSMAX)

    ! MPTABLE radiation block.
    call emit_array(n, 'ALBSAT', p%ALBSAT, 2)
    call emit_array(n, 'ALBDRY', p%ALBDRY, 2)
    call emit_array(n, 'ALBICE', p%ALBICE, 2)
    call emit_array(n, 'ALBLAK', p%ALBLAK, 2)
    call emit_array(n, 'OMEGAS', p%OMEGAS, 2)
    call emit(n, 'BETADS', 0, p%BETADS)
    call emit(n, 'BETAIS', 0, p%BETAIS)
    call emit_array(n, 'EG', p%EG, 2)

    ! MPTABLE global block.
    call emit(n, 'CO2',          0, p%CO2)
    call emit(n, 'O2',           0, p%O2)
    call emit(n, 'TIMEAN',       0, p%TIMEAN)
    call emit(n, 'FSATMX',       0, p%FSATMX)
    call emit(n, 'Z0SNO',        0, p%Z0SNO)
    call emit(n, 'SSI',          0, p%SSI)
    call emit(n, 'SNOW_RET_FAC', 0, p%SNOW_RET_FAC)
    call emit(n, 'SNOW_EMIS',    0, p%SNOW_EMIS)
    call emit(n, 'SWEMX',        0, p%SWEMX)
    call emit(n, 'TAU0',         0, p%TAU0)
    call emit(n, 'GRAIN_GROWTH', 0, p%GRAIN_GROWTH)
    call emit(n, 'EXTRA_GROWTH', 0, p%EXTRA_GROWTH)
    call emit(n, 'DIRT_SOOT',    0, p%DIRT_SOOT)
    call emit(n, 'BATS_COSZ',    0, p%BATS_COSZ)
    call emit(n, 'BATS_VIS_NEW', 0, p%BATS_VIS_NEW)
    call emit(n, 'BATS_NIR_NEW', 0, p%BATS_NIR_NEW)
    call emit(n, 'BATS_VIS_AGE', 0, p%BATS_VIS_AGE)
    call emit(n, 'BATS_NIR_AGE', 0, p%BATS_NIR_AGE)
    call emit(n, 'BATS_VIS_DIR', 0, p%BATS_VIS_DIR)
    call emit(n, 'BATS_NIR_DIR', 0, p%BATS_NIR_DIR)
    call emit(n, 'RSURF_SNOW',   0, p%RSURF_SNOW)
    call emit(n, 'RSURF_EXP',    0, p%RSURF_EXP)

    ! MPTABLE tile-drainage block (read unconditionally; opt_tdrn = 0 leaves it unused).
    call emit(n, 'KLAT_FAC',  0, p%KLAT_FAC)
    call emit(n, 'TDSMC_FAC', 0, p%TDSMC_FAC)
    call emit(n, 'TD_DC',     0, p%TD_DC)
    call emit(n, 'TD_DCOEF',  0, p%TD_DCOEF)
    call emit(n, 'TD_RADI',   0, p%TD_RADI)
    call emit(n, 'TD_SPAC',   0, p%TD_SPAC)
    call emit(n, 'TD_DDRAIN', 0, p%TD_DDRAIN)
    call emit_int(n, 'TD_DEPTH', 0, p%TD_DEPTH)
    call emit(n, 'TD_ADEPTH', 0, p%TD_ADEPTH)
    call emit_int(n, 'DRAIN_LAYER_OPT', 0, p%DRAIN_LAYER_OPT)
    call emit(n, 'TD_D',      0, p%TD_D)

    ! SOILPARM block.
    call emit_array(n, 'BEXP',   p%BEXP,   nsoil)
    call emit_array(n, 'DKSAT',  p%DKSAT,  nsoil)
    call emit_array(n, 'DWSAT',  p%DWSAT,  nsoil)
    call emit_array(n, 'PSISAT', p%PSISAT, nsoil)
    call emit_array(n, 'QUARTZ', p%QUARTZ, nsoil)
    call emit_array(n, 'SMCDRY', p%SMCDRY, nsoil)
    call emit_array(n, 'SMCMAX', p%SMCMAX, nsoil)
    call emit_array(n, 'SMCREF', p%SMCREF, nsoil)
    call emit_array(n, 'SMCWLT', p%SMCWLT, nsoil)
    call emit(n, 'F1',     0, p%F1)
    call emit(n, 'REFDK',  0, p%REFDK)
    call emit(n, 'REFKDT', 0, p%REFKDT)
    call emit(n, 'BVIC',   0, p%BVIC)
    call emit(n, 'AXAJ',   0, p%AXAJ)
    call emit(n, 'BXAJ',   0, p%BXAJ)
    call emit(n, 'XXAJ',   0, p%XXAJ)
    call emit(n, 'BDVIC',  0, p%BDVIC)
    call emit(n, 'GDVIC',  0, p%GDVIC)
    call emit(n, 'BBVIC',  0, p%BBVIC)

    ! GENPARM block plus the two derived quantities WRF forms in the transfer.
    call emit(n, 'CSOIL', 0, p%CSOIL)
    call emit(n, 'ZBOT',  0, p%ZBOT)
    call emit(n, 'CZIL',  0, p%CZIL)
    call emit(n, 'SLOPE', 0, p%SLOPE)
    call emit(n, 'KDT',   0, p%KDT)
    call emit(n, 'FRZX',  0, p%FRZX)
  end do

  close(unit)

contains

  subroutine emit(icase, field, index, value)
    integer, intent(in) :: icase, index
    character(len=*), intent(in) :: field
    real, intent(in) :: value
    write(unit, '(*(g0,:,","))') trim(names(icase)), vegtypes(icase), &
        soiltypes(icase), slopetypes(icase), soilcolors(icase), field, &
        index, value
  end subroutine emit

  subroutine emit_int(icase, field, index, value)
    integer, intent(in) :: icase, index, value
    character(len=*), intent(in) :: field
    write(unit, '(*(g0,:,","))') trim(names(icase)), vegtypes(icase), &
        soiltypes(icase), slopetypes(icase), soilcolors(icase), field, &
        index, value
  end subroutine emit_int

  subroutine emit_array(icase, field, values, count)
    integer, intent(in) :: icase, count
    character(len=*), intent(in) :: field
    real, intent(in) :: values(count)
    integer :: k
    do k = 1, count
      call emit(icase, field, k, values(k))
    end do
  end subroutine emit_array

end program run_noahmp_parameters_oracle
