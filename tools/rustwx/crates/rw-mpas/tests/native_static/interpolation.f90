! Oracle calls pristine MPAS: mpas_init_atm_hinterp.F:151-228.
program native_static_interpolation
 use init_atm_hinterp
 implicit none
 integer:: mode,i,j,q,bits
 integer:: methods(5)=[FOUR_POINT,W_AVERAGE4,W_AVERAGE16,SEARCH,0]
 real:: a(-2:13,-2:13,1),x,y,v
 do mode=0,5
  do j=-2,13
   do i=-2,13
    a(i,j,1)=real(11+i*17+j*23)*0.0137
    if (mode==1.and.modulo(i+2*j,3)==0) a(i,j,1)=0
    if (mode==2.and.i>=2.and.i<=9.and.j>=2.and.j<=9) a(i,j,1)=0
    if (mode==3) a(i,j,1)=0
    if (mode==4.and.modulo(i+j,2)==0) a(i,j,1)=0
    if (mode==5.and.(i/=0.or.j/=0)) a(i,j,1)=0
   end do
  end do
  do q=0,1023
   x=real(modulo(q*37,1550))*0.01-2.25
   y=real(modulo(q*59,1550))*0.01-2.25
   v=interp_sequence(x,y,1,a,-2,13,-2,13,1,1,0.,methods,1)
   bits=transfer(v,bits)
   write(*,'(4(i0,1x),z8.8)') mode,q,transfer(x,bits),transfer(y,bits),bits
  end do
 end do
end program
