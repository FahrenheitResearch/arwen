module ccpp_kind_types
  implicit none
  integer, parameter :: kind_phys = kind(1.0)
end module ccpp_kind_types

module module_model_constants
  use ccpp_kind_types, only: kind_phys
  implicit none
  real(kind_phys), parameter :: g = 9.81
  real(kind_phys), parameter :: r_d = 287.0
  real(kind_phys), parameter :: cp = 7.0 * r_d / 2.0
  real(kind_phys), parameter :: r_v = 461.6
  real(kind_phys), parameter :: cpv = 4.0 * r_v
  real(kind_phys), parameter :: cliq = 4190.0
  real(kind_phys), parameter :: cice = 2106.0
  real(kind_phys), parameter :: rcp = r_d / cp
  real(kind_phys), parameter :: p1000mb = 100000.0
  real(kind_phys), parameter :: rvovrd = r_v / r_d
  real(kind_phys), parameter :: xls = 2.85e6
  real(kind_phys), parameter :: xlv = 2.5e6
  real(kind_phys), parameter :: xlf = 3.50e5
  real(kind_phys), parameter :: svp1 = 0.6112
  real(kind_phys), parameter :: svp2 = 17.67
  real(kind_phys), parameter :: svp3 = 29.65
  real(kind_phys), parameter :: svpt0 = 273.15
  real(kind_phys), parameter :: ep_2 = r_d / r_v
  real(kind_phys), parameter :: p608 = rvovrd - 1.0
  real(kind_phys), parameter :: karman = 0.4
end module module_model_constants
