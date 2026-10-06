! Service-only interfaces for the standalone HRRR fork (WRFV3.9) advection
! module.  No boundary, advection, limiter, or physical constant is
! implemented here.  The fork's own frame/module_wrf_error.F needs ESMF, so
! its logging surface is stood in for by inert routines below.
module module_configure
  implicit none
  type grid_config_rec_type
    integer :: h_mom_adv_order = 5, v_mom_adv_order = 3
    integer :: h_sca_adv_order = 5, v_sca_adv_order = 3
    integer :: scalar_adv_opt = 0
    logical :: periodic_x = .false., periodic_y = .false.
    logical :: specified = .false., nested = .false.
    logical :: open_xs = .false., open_xe = .false.
    logical :: open_ys = .false., open_ye = .false.
    logical :: symmetric_xs = .false., symmetric_xe = .false.
    logical :: symmetric_ys = .false., symmetric_ye = .false.
    logical :: polar = .false.
  end type grid_config_rec_type
end module module_configure

! module_advect_em imports grid_config_rec_type through module_bc.
! None of module_bc's executable routines is called by this oracle.
module module_bc
  use module_configure, only: grid_config_rec_type
  implicit none
end module module_bc

module module_wrf_error
  implicit none
  character(len=512) :: wrf_err_message
contains
  logical function wrf_at_debug_level(level)
    integer, intent(in) :: level
    wrf_at_debug_level = .false.
  end function wrf_at_debug_level
end module module_wrf_error

subroutine wrf_abort()
  implicit none
  error stop 'WRF fatal error reached in advection oracle'
end subroutine wrf_abort

subroutine wrf_error_fatal(message)
  implicit none
  character(*), intent(in) :: message
  write(*, '(a)') trim(message)
  error stop 'WRF fatal error reached in advection oracle'
end subroutine wrf_error_fatal

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_message(message)
  implicit none
  character(*), intent(in) :: message
end subroutine wrf_message
