! Minimal stand-ins for the two WRF modules GSD MYNN v4.1 (NOAA-EMC WRF 3.9
! branch, tag v4.1.21) phys/module_bl_mynn.F uses.  The constants are the
! ARW (non-NMM) values of that branch's share/module_model_constants.F.
module module_model_constants
  implicit none
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.
  real, parameter :: cp = 7.*r_d/2.
  real, parameter :: r_v = 461.6
  real, parameter :: cpv = 4.*r_v
  real, parameter :: cliq = 4190.
  real, parameter :: cice = 2106.
  real, parameter :: rcp = r_d/cp
  real, parameter :: p1000mb = 100000.
  real, parameter :: rvovrd = r_v/r_d
  real, parameter :: xls = 2.85e6
  real, parameter :: xlv = 2.5e6
  real, parameter :: xlf = 3.50e5
  real, parameter :: svp1 = 0.6112
  real, parameter :: svp2 = 17.67
  real, parameter :: svp3 = 29.65
  real, parameter :: svpt0 = 273.15
  real, parameter :: ep_1 = r_v/r_d - 1.
  real, parameter :: ep_2 = r_d/r_v
  real, parameter :: karman = 0.4
end module module_model_constants

module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer, parameter :: p_qc = 2, p_qr = 3, p_qi = 4, p_qs = 5, p_qg = 6
  integer, parameter :: p_qnc = 7, p_qni = 8
end module module_state_description
