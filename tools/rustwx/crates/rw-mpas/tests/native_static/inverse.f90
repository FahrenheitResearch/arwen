program inverse
 use atm_advection
 implicit none
 integer::q,i,j,index(6)
 real::a(6,6),b(6,6)
 ! Keep all five searched pivots defined, then exercise native division by
 ! a zero final pivot. This is IEEE arithmetic, not an unset pivot index.
 do q=0,31
  a=0.0
  do i=1,5
   a(i,i)=real(i+q+1)
   do j=i+1,6
    a(i,j)=real(mod(i*7+j*11+q,13)-6)*0.125
   enddo
  enddo
  call MIGS(a,6,b,index)
  do i=1,6
   do j=1,6
    write(*,'(3(I4,1X),Z8.8)')q,i,j,transfer(b(i,j),0)
   enddo
  enddo
 enddo
end program
