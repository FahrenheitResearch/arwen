program geoindex
 use mpas_parse_geoindex
 use mpas_pool_routines
 implicit none
 type(mpas_pool_type)::pool
 integer::u,v,out,stat,q,i,err
 character(512)::line,argument,unreadable
 call get_command_argument(1,argument)
 ! MPAS mpas_parse_geoindex.F:72-88: nonexistent and unreadable owned files.
 call get_command_argument(2,unreadable)
 err=mpas_parse_index('native-missing.index',pool)
 if(err/=-1)stop 10
 if(len_trim(unreadable)>0)then
  err=mpas_parse_index(trim(unreadable),pool)
  if(err/=-1)stop 11
 endif
 open(newunit=u,file=trim(argument),status='old')
 open(newunit=out,file='geoindex.words',status='replace')
 q=0
 do
  open(newunit=v,file='synthetic.index',status='replace')
  do
   read(u,'(a)',iostat=stat)line
   if(stat/=0.or.line=='===')exit
   write(v,'(a)')trim(line)
  enddo
  close(v)
  if(stat/=0)exit
  q=q+1;count=0
  err=mpas_parse_index('synthetic.index',pool)
  write(out,'(I4,1X,I3)')q,err
  if(err==0)then
   do i=1,count
    select case(kinds(i))
    case('C')
    write(out,'(a,1X,a,1X,a)')trim(names(i)),kinds(i),trim(chars(i))
    case('R')
    write(out,'(a,1X,a,1X,Z8.8)')trim(names(i)),kinds(i),transfer(reals(i),0)
    case('I')
    write(out,'(a,1X,a,1X,I0)')trim(names(i)),kinds(i),ints(i)
    end select
   enddo
  endif
  write(out,'(a)')'==='
 enddo
 close(u);close(out)
end program
