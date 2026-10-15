! WRF public-domain cal_cldfra2, extracted verbatim by build.sh.
program run_cldfra2
implicit none
integer, parameter :: nc=64, nk=4
real :: qc(nc,nk,1), qi(nc,nk,1), cf(nc,nk,1), thresh, lo, hi
integer :: i,k,a,b,flags(2),u
logical :: fqc,fqi
thresh=1.e-6
lo=nearest(thresh,-1.0); hi=nearest(thresh,1.0)
open(newunit=u,file='columns.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
write(u) nc,nk
! Both routine and no-call driver states, all four moisture memberships.
do b=0,1
 do a=0,3
  fqc=btest(a,0); fqi=btest(a,1)
  flags=[merge(1,0,fqc),merge(1,0,fqi)]
  do k=1,nk
   do i=1,nc
    select case(mod(i+k,8))
    case(0); qc(i,k,1)=0.; qi(i,k,1)=2.e-6
    case(1); qc(i,k,1)=lo; qi(i,k,1)=0.
    case(2); qc(i,k,1)=thresh; qi(i,k,1)=0.
    case(3); qc(i,k,1)=hi; qi(i,k,1)=0.
    case(4); qc(i,k,1)=0.5*thresh; qi(i,k,1)=0.5*thresh
    case(5); qc(i,k,1)=0.5*hi; qi(i,k,1)=0.5*hi
    case(6); qc(i,k,1)=-thresh; qi(i,k,1)=2.*thresh
    case(7); qc(i,k,1)=tiny(thresh)*0.5; qi(i,k,1)=tiny(thresh)*0.5
    end select
   end do
  end do
  cf=0.375
  if (b==0) then
   call cal_cldfra2(cf,qc,qi,fqc,fqi,1,nc+1,1,2,1,nk+1,1,nc,1,1,1,nk,1,nc,1,1,1,nk)
  else
   ! Driver initializes CLDFRA first; it skips the call for neither flag.
   cf=0.
   if (fqc.or.fqi) call cal_cldfra2(cf,qc,qi,fqc,fqi,1,nc+1,1,2,1,nk+1,1,nc,1,1,1,nk,1,nc,1,1,1,nk)
  end if
  write(u) flags,b,qc,qi,cf
 end do
end do
close(u)
end program
