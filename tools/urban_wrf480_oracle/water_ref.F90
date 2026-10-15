! ABI-only wrapper for the release SRT and SMFLX, with four layers.
! Only the assigned first three tendency/matrix/moisture slots are returned.
subroutine water_ref(out,smcp,zsoil,et,root,v) bind(C)
  use iso_c_binding
  use module_sf_urban, only: srt, smflx
  implicit none
  real(c_float), intent(out) :: out(22)
  real(c_float), intent(in) :: smcp(4), zsoil(4), et(4), root(4), v(13)
  real :: rh(4), a(4), b(4), c(4), smc(4), cmc, r1, r2, r3, drip
  rh=0.;a=0.;b=0.;c=0.;smc=smcp
  call srt(rh,v(10),et,smcp,4,v(13),zsoil,v(7),v(6),v(3),v(4),r1,r2,v(1),v(5),root,a,b,c)
  out(1:3)=rh(1:3);out(4:6)=a(1:3);out(7:9)=b(1:3);out(10:12)=c(1:3)
  out(21)=r1;out(22)=r2
  call smflx(smcp,smc,4,v(12),cmc,v(1),v(2),zsoil,v(3),v(4),v(5),root,v(6), &
             v(7),v(8),v(9),r1,r2,r3,v(10),v(11),et,drip)
  out(13:15)=smc(1:3)
  out(16)=cmc;out(17)=r1;out(18)=r2;out(19)=r3;out(20)=drip
end subroutine water_ref
