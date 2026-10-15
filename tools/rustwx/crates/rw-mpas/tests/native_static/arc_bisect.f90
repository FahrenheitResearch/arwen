! Calls the complete pristine native atm_advection module, single RKIND.
program arc_words
 use atm_advection
 implicit none
 integer::q,i,word,kindbits(7)
 real::a(3),b(3),c(3),tiny
 kindbits=[int(z'00800000'),int(z'01000000'),int(z'00000002'),int(z'1B800000'),int(z'20000000'),int(z'23800000'),int(z'2B800000')]
 do q=0,223
  tiny=transfer(kindbits(mod(q,7)+1),tiny)
  a=[real(q/7+1),0.0,0.0]
  b=[-a(1),tiny,tiny*real(mod(q,3))]
  call arc_bisect(a(1),a(2),a(3),b(1),b(2),b(3),c(1),c(2),c(3))
  write(*,'(i0,1x,9(z8.8,1x))')q,(transfer(a(i),word),i=1,3),(transfer(b(i),word),i=1,3),(transfer(c(i),word),i=1,3)
 enddo
end program
