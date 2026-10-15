 open(newunit=u,file='climatology.words',status='replace')
 do q=1,nCells
  write(u,'(I4,1X,2(Z8.8,1X),I1,1X,29(Z8.8,1X))')q,transfer(latCell(q),0),transfer(lonCell(q),0),landmask(q), &
   (transfer(greenfrac(k,q),0),k=1,12),(transfer(albedo12m(k,q),0),k=1,12), &
   transfer(snoalb(q),0),transfer(soiltemp(q),0),transfer(shdmin(q),0),transfer(shdmax(q),0)
 enddo
 close(u)
contains
 subroutine mpas_f_to_c_string(f,c)
 character(*)::f
 character(c_char)::c(:)
 integer::i
 do i=1,len_trim(f)
  c(i)=f(i:i)
 enddo
 c(len_trim(f)+1)=c_null_char
 end subroutine
 subroutine init_atm_check_read_error(status,fname)
 integer::status
 character(*)::fname
 if(status/=0)then
  write(*,*)'geogrid read error',status,trim(fname)
  stop 3
 endif
 end subroutine
end program
