module optics_module
  use module_data_rrtmgaeropt
  implicit none
  integer, parameter :: lunerr=-1
contains
#include "prep.inc"
#include "mie.inc"
subroutine prepare_gocart(chem,alt,relhum,radius_core,radius_wet,number_bin,swrefindx,swrefindx_core,swrefindx_shell, &
 lwrefindx,lwrefindx_core,lwrefindx_shell,nz)
 use module_configure,only:num_chem
 integer :: nz
 integer,parameter :: nbin_o=9,uoc_flag=0,ids=1,ide=1,jds=1,jde=1,ims=1,ime=1,jms=1,jme=1, &
 its=1,ite=1,jts=1,jte=1,kds=1,kms=1,kts=1
 integer :: kde,kme,kte
 real :: chem(1,nz,1,num_chem),alt(1,nz,1),relhum(1,nz,1)
 real :: radius_core(1,nz,1,9),radius_wet(1,nz,1,9),number_bin(1,nz,1,9)
 complex :: swrefindx(1,nz,1,9,4),swrefindx_core(1,nz,1,9,4),swrefindx_shell(1,nz,1,9,4)
 complex :: lwrefindx(1,nz,1,9,16),lwrefindx_core(1,nz,1,9,16),lwrefindx_shell(1,nz,1,9,16)
 kde=nz;kme=nz;kte=nz
 include 'prep_call.inc'
end subroutine
subroutine evaluate_mie(number_bin_col,radius_wet_col,swrefindx_col,lwrefindx_col,dz,swsizeaer,swextaer,swwaer, &
 swgaer,swtauaer,lwextaer,lwtauaer,l2,l3,l4,l5,l6,l7,swbscoef,nz)
 integer :: nz,kte
 integer,parameter :: id=1,iclm=1,jclm=1,nbin_o=9,kts=1,option_mie=1
 real(8),parameter :: curr_secs=0.d0
 real :: number_bin_col(9,nz),radius_wet_col(9,nz),dz(nz)
 complex :: swrefindx_col(9,nz,4),lwrefindx_col(9,nz,16)
 real :: swsizeaer(4,nz),swextaer(4,nz),swwaer(4,nz),swgaer(4,nz),swtauaer(4,nz),swbscoef(4,nz)
 real :: lwextaer(16,nz),lwtauaer(16,nz),l2(4,nz),l3(4,nz),l4(4,nz),l5(4,nz),l6(4,nz),l7(4,nz)
 kte=nz
 include 'mie_call.inc'
end subroutine
subroutine clamp_outputs(swtauaer,swextaer,swwaer,swgaer,swbscoef,lwtauaer,lwextaer,nz)
 integer :: nz,k,ns
 integer,parameter :: nspint=4,iclm=1,jclm=1
 real :: swtauaer(4,nz),swextaer(4,nz),swwaer(4,nz),swgaer(4,nz),swbscoef(4,nz)
 real :: lwtauaer(16,nz),lwextaer(16,nz)
 real :: tauaersw(1,nz,1,4),extaersw(1,nz,1,4),waersw(1,nz,1,4),gaersw(1,nz,1,4),bscoefsw(1,nz,1,4)
 real :: tauaerlw(1,nz,1,16),extaerlw(1,nz,1,16)
 do k=1,nz
  include 'sw_clamps.inc'
  include 'lw_clamps.inc'
 enddo
 swtauaer=transpose(tauaersw(1,:,1,:));swextaer=transpose(extaersw(1,:,1,:))
 swwaer=transpose(waersw(1,:,1,:));swgaer=transpose(gaersw(1,:,1,:));swbscoef=transpose(bscoefsw(1,:,1,:))
 lwtauaer=transpose(tauaerlw(1,:,1,:));lwextaer=transpose(extaerlw(1,:,1,:))
end subroutine
end module
