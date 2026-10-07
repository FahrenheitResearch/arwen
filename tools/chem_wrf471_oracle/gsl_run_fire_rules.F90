program gsl_run_fire_rules
 use machine
 use oracle_io
 implicit none
 integer,parameter :: nc=30,its=1,ite=nc,jts=1,jte=1,ebb_dcycle=2
 integer :: i,j,g,e,n
 integer :: fire_type(nc,1),categories(3)
 real(kind_phys) :: vegfrac(nc,20,1),xlat(nc,1),xlong(nc,1),ebu_in(nc,1)
 real(kind_phys) :: lu_nofire(nc,1),lu_qfire(nc,1),lu_sfire(nc,1),thresholds(3),value
 real(kind_phys),parameter :: ebb_min=1.e-3
 character(1024) :: root
 call oracle_root(root)
 vegfrac=0.;xlat=0.;xlong=0.;ebu_in=1.
 categories=(/11,12,8/);thresholds=(/.95,.9,.8/)
 n=0
 do g=1,3
  do e=-1,1
   n=n+1;value=thresholds(g)
   if(e/=0) value=nearest(value,real(e,kind_phys))
   vegfrac(n,categories(g),1)=value
  enddo
 enddo
 do e=-1,1
  n=n+1;xlat(n,1)=33.;xlong(n,1)=260.
  if(e/=0) xlong(n,1)=nearest(xlong(n,1),real(e,kind_phys))
 enddo
 do g=1,2
  do e=-1,1
   n=n+1;xlong(n,1)=261.;xlat(n,1)=merge(25._kind_phys,41._kind_phys,g==1)
   if(e/=0) xlat(n,1)=nearest(xlat(n,1),real(e,kind_phys))
  enddo
 enddo
 ! Multiple terms exercise source association and rule precedence.
 vegfrac(19,11,1)=.3;vegfrac(19,15,1)=.3;vegfrac(19,17,1)=.2;vegfrac(19,20,1)=.16
 vegfrac(20,12,1)=.5;vegfrac(20,13,1)=.42
 vegfrac(21,8,1)=.3;vegfrac(21,9,1)=.3;vegfrac(21,10,1)=.21
 ebu_in(22,1)=nearest(ebb_min,-1._kind_phys)
 ebu_in(23,1)=ebb_min
 ebu_in(24,1)=nearest(ebb_min,1._kind_phys)
 lu_nofire=-huge(1._kind_phys);lu_qfire=huge(1._kind_phys);lu_sfire=-huge(1._kind_phys)
 include 'fire_rules.inc'
 call oracle_open('edges')
 call oracle_put('fractions',real(vegfrac,4));call oracle_put('latitude',real(xlat,4))
 call oracle_put('longitude',real(xlong,4));call oracle_put('emission',real(ebu_in,4))
 call oracle_put('minimum',real(ebb_min,4));call oracle_put('fire_type',fire_type)
 call oracle_close()
end program
