program run_sounding_controls
 use sfire_ideal_atmos_oracle
 use oracle_io
 implicit none
 integer,parameter::nmax=1000
 real::z(nmax),pm(nmax),pd(nmax),th(nmax),rho(nmax),u(nmax),v(nmax),q(nmax)
 real::raw_z(nmax),raw_th(nmax),raw_q(nmax),raw_u(nmax),raw_v(nmax),surface(3)
 integer::nl,raw_nl,dryflag
 character(4096)::root
 character(60)::name
 call oracle_root(root)
 call read_sounding(surface(1),surface(2),surface(3),raw_z,raw_th,raw_q,raw_u,raw_v,nmax,raw_nl,.false.)
 do dryflag=0,1
  call get_sounding(z,pm,pd,th,rho,u,v,q,dryflag==1,nmax,nl)
  write(name,'(a,i0)')'sounding_dry',dryflag
  call oracle_open(trim(name))
  call oracle_put('height',z(1:nl));call oracle_put('p_moist',pm(1:nl));call oracle_put('p_dry',pd(1:nl))
  call oracle_put('theta',th(1:nl));call oracle_put('rho',rho(1:nl));call oracle_put('u',u(1:nl))
  call oracle_put('v',v(1:nl));call oracle_put('qv',q(1:nl))
  call oracle_put('input_height',raw_z(1:raw_nl));call oracle_put('input_theta',raw_th(1:raw_nl))
  call oracle_put('input_qv',raw_q(1:raw_nl));call oracle_put('input_u',raw_u(1:raw_nl))
  call oracle_put('input_v',raw_v(1:raw_nl));call oracle_put('input_surface',surface)
  call oracle_close()
 enddo
end program
