! Declarations-only stand-ins for the registry-generated WRF modules the
! GOCART (chem_opt=300) sources USE, plus service-only error hooks.
!
! WRF generates module_state_description and module_configure from the
! Registry at build time, so they are not files of the WRF tree and cannot be
! compiled from it.  Everything below is either an index or switch constant
! with the value the Registry gives it for the gocart_simple package
! (Registry/registry.chem:4022, chem array in package order after the
! PARAM_FIRST_SCALAR=2 dummy), or a service routine that aborts or does
! nothing.  Nothing here computes a physical quantity.
!
! A species absent from the active package gets index 1 in WRF (below
! PARAM_FIRST_SCALAR), which is what the IF (p_x >= param_first_scalar) tests
! in the chem code rely on; the absent names below carry 1 for that reason.
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  ! chem, gocart_simple package order (registry.chem:4022)
  integer, parameter :: p_so2 = 2, p_sulf = 3, p_dms = 4, p_msa = 5,     &
                        p_p25 = 6, p_bc1 = 7, p_bc2 = 8, p_oc1 = 9,       &
                        p_oc2 = 10, p_dust_1 = 11, p_dust_2 = 12,         &
                        p_dust_3 = 13, p_dust_4 = 14, p_dust_5 = 15,      &
                        p_seas_1 = 16, p_seas_2 = 17, p_seas_3 = 18,      &
                        p_seas_4 = 19, p_p10 = 20
  integer, parameter :: num_chem = 20
  ! species of other packages the GOCART files name but never reach here
  integer, parameter :: p_ac0 = 1, p_corn = 1, p_p25i = 1, p_p25j = 1,    &
                        p_soila = 1, p_naai = 1, p_naaj = 1, p_seas = 1,  &
                        p_h2o2 = 1, p_ho = 1, p_no3 = 1
  ! moist (Thompson-like set; only the indices are used)
  integer, parameter :: p_qv = 2, p_qc = 3, p_qr = 4, p_qi = 5,           &
                        p_qs = 6, p_qg = 7, num_moist = 7
  ! emis_dust (dustgocart / dustgocartafwa, registry.chem:4127-4128)
  integer, parameter :: p_edust1 = 2, p_edust2 = 3, p_edust3 = 4,         &
                        p_edust4 = 5, p_edust5 = 6, num_emis_dust = 6
  ! emis_seas (seasgocart, registry.chem:4133)
  integer, parameter :: p_eseas1 = 2, p_eseas2 = 3, p_eseas3 = 4,         &
                        p_eseas4 = 5, num_emis_seas = 5
  ! emis_ant, gocart_ecptec package (emiss_opt==6, registry.chem:4058)
  integer, parameter :: p_e_so2 = 2, p_e_sulf = 3, p_e_bc = 4, p_e_oc = 5, &
                        p_e_pm_25 = 6, p_e_pm_10 = 7, num_emis_ant = 7
  ! chem_opt package values (registry.chem package lines)
  integer, parameter :: gocart_simple = 300, gocartracm_kpp = 301,        &
                        gocartradm2 = 303, chem_vash = 400, dust = 401,   &
                        chem_volc = 402, mozcart_kpp = 112,               &
                        t1_mozcart_kpp = 114, cb05_sorg_aq_kpp = 131,     &
                        cb05_sorg_vbs_aq_kpp = 132
  ! sf_surface_physics package values (Registry.EM_COMMON:3156-3161)
  integer, parameter :: lsmscheme = 2, ruclsmscheme = 3, pxlsmscheme = 7
  ! dust_opt / seas_opt / dmsemis_opt values (registry.chem:4127-4136)
  integer, parameter :: dustgocart = 1, dustgocartafwa = 3, dustuoc = 4,  &
                        seasgocart = 1, dmsgocart = 1
end module module_state_description

module module_configure
  use module_state_description
  implicit none
  type grid_config_rec_type
    integer :: chem_opt = gocart_simple
    integer :: dust_opt = 0
    integer :: seas_opt = 0
    integer :: dmsemis_opt = 0
    integer :: emiss_opt = 0          ! registry.chem:3842
    integer :: kemit = 9              ! registry.chem:3780
    integer :: start_month = 1
    integer :: num_soil_layers = 4
    integer :: sf_surface_physics = 2
    ! AFWA switches (registry.chem, dust_* rconfig rows)
    integer :: dust_dsr = 0
    integer :: dust_smois = 0
    integer :: dust_soils = 0
    integer :: dust_veg = 0
    logical :: restart = .false.
  end type grid_config_rec_type
end module module_configure

! module_gocart_chem.F:14 USEs calc_zenith from module_phot_mad but never
! calls it on the GOCART_SIMPLE path; this satisfies the USE and aborts if
! anything ever did call it.
module module_phot_mad
  implicit none
contains
  subroutine calc_zenith()
    error stop 'stub calc_zenith reached: not part of the GOCART path'
  end subroutine calc_zenith
end module module_phot_mad

subroutine wrf_debug(level, str)
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_message(str)
  character(len=*), intent(in) :: str
  write(0, '(A)') trim(str)
end subroutine wrf_message

subroutine wrf_error_fatal(str)
  character(len=*), intent(in) :: str
  write(0, '(A)') trim(str)
  error stop 'wrf_error_fatal'
end subroutine wrf_error_fatal
