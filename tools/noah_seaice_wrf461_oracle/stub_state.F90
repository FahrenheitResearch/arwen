! module_state_description is generated from WRF's Registry at configure time
! and does not exist in a bare source tree.  seaice_noah reads exactly three of
! its package constants, to test sf_urban_physics against the urban schemes
! (phys/module_sf_noah_seaice_drv.F, the B_T_BEP/B_Q_BEP hand-over).  The
! values are WRF's Registry.EM_COMMON package numbers for sf_urban_physics.
! run_seaice.F90 passes sf_urban_physics = 0, so none of them is reached.
module module_state_description
  implicit none
  integer, parameter :: NOAHUCMSCHEME = 1
  integer, parameter :: BEPSCHEME = 2
  integer, parameter :: BEP_BEMSCHEME = 3
end module module_state_description
