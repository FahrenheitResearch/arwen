! Extend the shared registry declarations for the optics output indices.
#include "registry_declarations.inc"
module module_configure
  use module_state_description
  implicit none
  integer, parameter :: p_extcof3=1,p_extcof55=2,p_extcof106=3,p_extcof3_5=4,p_extcof8_12=5
  integer, parameter :: p_bscof3=1,p_bscof55=2,p_bscof106=3
  integer, parameter :: p_asympar3=1,p_asympar55=2,p_asympar106=3
end module
module module_data_sorgam
  implicit none
  include 'modal_parameters.inc'
  include 'hygro_parameters.inc'
end module
module module_data_mosaic_asect
  implicit none
  include 'msa_parameter.inc'
end module
module module_data_gocartchem
  implicit none
  include 'mass_parameters.inc'
end module
! Message services are the only executable stubs.
module module_peg_util
contains
  subroutine peg_message(unit,msg)
    integer :: unit
    character(*) :: msg
    print *,trim(msg)
  end subroutine
  subroutine peg_error_fatal(unit,msg)
    integer :: unit
    character(*) :: msg
    print *,trim(msg)
    error stop 'pinned WRF fatal'
  end subroutine
end module
