program run_stage_wvsat
 use shr_kind_mod, only: r8=>shr_kind_r8
 use module_cam_esinti, only: esinti
 use physconst, only: epsilo,latvap,latice,rh2o,cpair,tmelt
 use wv_saturation, only: estblf,aqsat,fqsatd
 use uwio
 implicit none
 integer,parameter :: cap=16000
 real(r8) :: ts(2000),t(1,cap),p(1,cap),es(1,cap),qs(1,cap),lookup(cap),fe(cap),fq(cap),gam(cap)
 real(r8) :: base,press(13),e1(1),q1(1),g1(1),tt(1),pp(1)
 integer :: nt,n,i,j,k,status(cap)
 character(512) :: stem
 call get_command_argument(1,stem)
 call esinti(epsilo,latvap,latice,rh2o,cpair,tmelt)
 nt=0
 do i=0,250
   call add(150._r8+real(i,r8))
 end do
 do i=0,203
   base=173.16_r8+real(i,r8)
   call add(nearest(base,-1._r8));call add(base);call add(nearest(base,1._r8))
 end do
 do i=0,80
   base=tmelt-20._r8+real(i,r8)*0.25_r8
   call add(nearest(base,-1._r8));call add(base);call add(nearest(base,1._r8))
 end do
 n=0
 do i=1,nt
   base=(1._r8-epsilo)*estblf(ts(i))
   press=[100._r8,1000._r8,10000._r8,50000._r8,100000._r8,110000._r8, &
       nearest(base,-1._r8),base,nearest(base,1._r8), &
       estblf(ts(i)),nearest(estblf(ts(i)),-1._r8),nearest(estblf(ts(i)),1._r8), &
       100._r8+109900._r8*real(i-1,r8)/real(nt-1,r8)]
   do j=1,13
     n=n+1;t(1,n)=ts(i);p(1,n)=press(j);lookup(n)=estblf(ts(i))
     tt(1)=ts(i);pp(1)=press(j)
     status(n)=fqsatd(tt,pp,e1,q1,g1,1)
     fe(n)=e1(1);fq(n)=q1(1);gam(n)=g1(1)
   end do
 end do
 call aqsat(t,p,es,qs,1,1,cap,1,n)
 call uwio_open(stem)
 call uwio_put_r8('t',t,n);call uwio_put_r8('p',p,n)
 call uwio_put_r8('estblf',lookup,n)
 call uwio_put_r8('aqsat_es',es,n);call uwio_put_r8('aqsat_qs',qs,n)
 call uwio_put_r8('fqsatd_es',fe,n);call uwio_put_r8('fqsatd_qs',fq,n)
 call uwio_put_r8('fqsatd_gam',gam,n);call uwio_put_i4('status',status,n)
 call uwio_close()
 print *, 'saturation records=',n
contains
 subroutine add(x)
 real(r8),intent(in)::x
 nt=nt+1;ts(nt)=x
 end subroutine
end program
