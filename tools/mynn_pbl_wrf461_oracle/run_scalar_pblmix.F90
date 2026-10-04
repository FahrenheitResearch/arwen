! Exercise the unmodified WRF v4.6.1 diff4d with its real scalar exclusions.
! The extracted WRF routines retain their original public-domain notice.
program run_scalar_pblmix
  use scalar_pblmix_wrf_oracle, only: diff4d
  implicit none
  integer, parameter :: nz=50, ns=11, ncase=8
  real :: dz(1,nz+1,1),rho(1,nz+1,1),kh(1,nz+1,1),km(1,nz+1,1)
  real :: scalar(1,nz+1,1,ns), tendency(1,nz+1,1,ns),dt,scale
  integer :: ic,k,im,unit
  character(len=512) :: path
  call get_command_argument(1,path)
  open(newunit=unit,file=trim(path),status='replace',action='write')
  write(unit,'(a)') 'case,species,k,dt,qn,dz,rho,exch_h,tendency'
  do ic=1,ncase
    dt=15.0
    if(ic.eq.7) dt=300.0
    if(ic.eq.8) dt=0.5
    do k=1,nz+1
      dz(1,k,1)=25.0+real(k*k)*0.75
      rho(1,k,1)=1.25/(1.0+real(k)*0.15)
      kh(1,k,1)=real(mod(k*19,47))*3.0
      if(ic.eq.1) kh(1,k,1)=0.0
      if(ic.eq.5) kh(1,k,1)=0.03125*real(k)
      if(ic.eq.6) kh(1,k,1)=5000.0
      km(1,k,1)=kh(1,k,1)*1.5
      do im=1,ns
        scale=1.0
        if(im.eq.1) scale=1.e7
        if(im.eq.2) scale=1.e4
        if(im.eq.3) scale=1.e9
        if(im.eq.4) scale=1.e6
        scalar(1,k,1,im)=scale*(0.25+real(mod(k*7,13))*0.0625)
        if(ic.eq.2) scalar(1,k,1,im)=0.0
        if(ic.eq.3) then
          scalar(1,k,1,im)=scale*0.03125
          if(k.le.3) scalar(1,k,1,im)=scale*8.0
        endif
        if(ic.eq.4) then
          scalar(1,k,1,im)=scale*0.03125
          if(k.ge.nz-2) scalar(1,k,1,im)=scale*8.0
        endif
      enddo
    enddo
    tendency=-765.25
    call diff4d(dt,dz,scalar,.true.,rho,kh,km,tendency,ns,1, &
                1,1,1,1,1,nz+1,1,1,1,1,1,nz+1,1,1,1,1,1,nz)
    do im=1,ns
      do k=1,nz
        write(unit,'(i0,",",i0,",",i0,6(",",es24.16))') &
          ic,im,k,dt,scalar(1,k,1,im),dz(1,k,1),rho(1,k,1), &
          kh(1,k,1),tendency(1,k,1,im)
      enddo
    enddo
  enddo
  close(unit)
end program run_scalar_pblmix
