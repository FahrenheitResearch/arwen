program cloud_optics_oracle
  use mo_rte_kind, only: wp
  use mo_cloud_optics_rrtmgp, only: ty_cloud_optics_rrtmgp
  use mo_optical_props, only: ty_optical_props_2str
  use mo_optics_utils_rrtmgp, only: load_cloud_optics
  implicit none

  integer, parameter :: ncol = 4, nlay = 1
  type(ty_cloud_optics_rrtmgp) :: cloud_spec
  type(ty_optical_props_2str) :: clouds
  real(wp) :: lwp(ncol, nlay), iwp(ncol, nlay)
  real(wp) :: rel(ncol, nlay), dei(ncol, nlay)
  character(len=512) :: coefficient_file
  character(len=16) :: kind
  character(len=128) :: error_msg
  integer :: icol, iband

  call get_command_argument(1, kind)
  call get_command_argument(2, coefficient_file)
  if (len_trim(kind) == 0 .or. len_trim(coefficient_file) == 0) &
    error stop "usage: cloud_optics_oracle KIND CLOUD_COEFFICIENT_FILE"

  ! Four spot columns: liquid, ice, mixed phase, and clear.
  lwp(:, 1) = [12.0_wp, 0.0_wp, 4.25_wp, 0.0_wp]
  iwp(:, 1) = [0.0_wp, 8.5_wp, 6.75_wp, 0.0_wp]
  rel(:, 1) = [7.125_wp, 10.0_wp, 14.375_wp, 10.0_wp]
  dei(:, 1) = [50.0_wp, 57.5_wp, 123.75_wp, 50.0_wp]

  ! Same loader, medium ice roughness, allocation, and cloud-optics call used
  ! by examples/all-sky/rrtmgp_allsky.F90.
  call load_cloud_optics(cloud_spec, trim(coefficient_file))
  error_msg = cloud_spec%set_ice_roughness(2)
  call check(error_msg)
  error_msg = clouds%init(cloud_spec)
  call check(error_msg)
  error_msg = clouds%alloc_2str(ncol, nlay)
  call check(error_msg)
  error_msg = cloud_spec%cloud_optics(lwp, iwp, rel, dei, clouds)
  call check(error_msg)

  write(*, '(A)') "kind,column,lwp,iwp,reliq,dgice,band,tau,ssa,g"
  do icol = 1, ncol
    do iband = 1, size(clouds%tau, 3)
      write(*, '(A,",",I0,",",4(ES24.16E3,","),I0,",",2(ES24.16E3,","),ES24.16E3)') &
        trim(kind), icol - 1, lwp(icol, 1), iwp(icol, 1), &
        rel(icol, 1), dei(icol, 1), iband - 1, &
        clouds%tau(icol, 1, iband), clouds%ssa(icol, 1, iband), &
        clouds%g(icol, 1, iband)
    end do
  end do

contains
  subroutine check(message)
    character(len=*), intent(in) :: message
    if (len_trim(message) /= 0) then
      write(*, '(A)') trim(message)
      error stop 1
    end if
  end subroutine check
end program cloud_optics_oracle
