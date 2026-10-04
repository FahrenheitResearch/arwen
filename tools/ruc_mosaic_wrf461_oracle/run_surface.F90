! WRF v4.6.1 public-domain module, called without source changes.
program run_mosaic_surface
  use module_sf_ruclsm, only: ruclsm_soilvegparm, soilvegin
  implicit none
  integer, parameter :: nl=21, ns=19
  integer :: unit, n, lu, so, forest, soil, veg, k
  real :: lf(nl), sf(ns), tmin, tmax, green, z0, lai, incoming_z0, incoming_lai
  real :: emiss, pc, qwrtz, rhocs, bclh, dqm, ksat, psis, qmin, ref, wilt
  logical :: keep_lai
  character(len=1024) :: output
  call get_command_argument(1,output)
  call ruclsm_soilvegparm('MODI-RUC','STAS-RUC')
  open(newunit=unit,file=trim(output),status='replace')
  write(unit,'(A)') 'case,mosaic_lu,mosaic_soil,isltyp,ivgtyp,shdmin,shdmax,vegfrac,znt_before,lai_before,rdlai2d,iforest,emiss,pc,znt,lai,qwrtz,rhocs,bclh,dqm,ksat,psis,qmin,ref,wilt'
  do n=1,12
    lu=1; so=1; soil=4; veg=12
    tmin=10.; tmax=90.; green=85.
    incoming_z0=0.00037; incoming_lai=2.345
    keep_lai=n==4
    lf=0.; sf=0.
    lf(1)=.15; lf(12)=.55; lf(14)=.3
    sf(6)=.35; sf(8)=.5; sf(14)=.15
    if(n==2) green=10.
    if(n==3) then
      tmax=10.5; green=10.25
    endif
    if(n==5) lf=lf*.5
    if(n==6) lf=lf*1.1
    if(n==7) then
      sf=0.; sf(14)=1.
    endif
    if(n==8) sf=sf*1.1
    if(n==9) then
      lf=0.; lf(17)=1.; sf=0.; sf(14)=1.
      soil=14; veg=17
    endif
    if(n==10) then
      lu=0
    endif
    if(n==11) then
      so=0
    endif
    if(n==12) then
      lf=0.; lf(12)=1.; sf=0.; sf(4)=1.
    endif
    z0=incoming_z0; lai=incoming_lai
    call soilvegin(lu,so,sf,ns,tmin,tmax,nl,veg,soil,17,.false.,forest, &
        lf,green,emiss,pc,z0,lai,keep_lai,qwrtz,rhocs,bclh,dqm,ksat,psis,qmin,ref,wilt,n,1)
    write(unit,'(*(g0,:,","))') n,lu,so,soil,veg,tmin,tmax,green,incoming_z0,incoming_lai, &
        merge(1,0,keep_lai),forest,emiss,pc,z0,lai,qwrtz,rhocs,bclh,dqm,ksat,psis,qmin,ref,wilt
  enddo
  close(unit)
end program
