program polynomial
 use atm_advection
 implicit none
 integer::q,i,j,edges
 real::a(25,25),b(25,25),w(25,25),x,y,t,r
 do q=0,255
  edges=5+mod(q,4)
  a=0.0;w=0.0
  a(1,1)=1.0;w(1,1)=1.0
  do i=1,edges
   t=real(i-1)*6.283185307179586/real(edges)+real(q)*0.013
   r=1000.0+real(mod(q*17+i*31,700))
   x=cos(t)*r;y=sin(t)*r
   a(i+1,1:6)=[1.0,x,y,x*x,x*y,y*y]
   w(i+1,i+1)=1.0
  enddo
  call poly_fit_2(a,b,w,edges+1,6,25)
  do i=1,6
   do j=1,edges+1
    write(*,'(3(I4,1X),Z8.8)')q,i,j,transfer(b(i,j),0)
   enddo
  enddo
 enddo
end program
