! Service-only interfaces used by the standalone WRF advection module.
! No boundary, advection, limiter, or physical constant is implemented here.
module module_configure
  implicit none
  type grid_config_rec_type
    integer :: h_mom_adv_order = 5, v_mom_adv_order = 3
    integer :: h_sca_adv_order = 5, v_sca_adv_order = 3
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

subroutine wrf_abort()
  implicit none
  error stop 'WRF fatal error reached in advection oracle'
end subroutine wrf_abort

subroutine wrf_debug(level,message)
  implicit none
  integer, intent(in) :: level
  character(*), intent(in) :: message
end subroutine wrf_debug
