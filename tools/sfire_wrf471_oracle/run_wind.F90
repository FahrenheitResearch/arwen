program run_wind
  use module_sfire_wind_oracle
  use module_fr_fire_util
  use module_model_constants, only: g
  use oracle_io
  implicit none
  integer, parameter :: nx=9,ny=7,nz=6,h=2,maxrx=4,maxry=3,fh=3
  integer, parameter :: ml=1-h,mh=nx+h,mb=1-h,mt=ny+h
  real :: u(ml:mh,1:nz+1,mb:mt),v(ml:mh,1:nz+1,mb:mt)
  real :: ph(ml:mh,1:nz+1,mb:mt),phb(ml:mh,1:nz+1,mb:mt)
  real :: rough(ml:mh,mb:mt),terrain(ml:mh,mb:mt),ua(ml:mh,mb:mt),va(ml:mh,mb:mt)
  real, allocatable :: uf(:,:),vf(:,:),frough(:,:)
  integer :: i,j,k,m,rx,ry,fl,fr,fb,ft,nfx,nfy,ix,jy,tl,th,tb,tt,lft,rft,bft,tft
  integer :: bounds(4),fbounds(4),atiles(4,4),ftiles(4,4)
  real :: height,z,ref
  character(len=1024) :: root
  character(len=64) :: name
  call oracle_root(root)
  fire_print_msg=0
  fire_lsm_zcoupling_ref=50.
  do j=mb,mt
    do i=ml,mh
      terrain(i,j)=340.+0.19*i*i+0.53*j*j+0.13*i*j
      rough(i,j)=0.04+0.035*modulo(i+3*j,8)
      if(i==4 .and. j==4) rough(i,j)=11.
      do k=1,nz+1
        u(i,k,j)=2.3+0.071*i*i-0.13*j+0.45*k+0.008*i*j*k
        v(i,k,j)=-1.7+0.019*j*j+0.17*i-0.21*k+0.006*i*j*k
      enddo
      do k=1,nz+1
        z=terrain(i,j)+(7.+0.03*i+0.07*j)*real(k-1)**2
        phb(i,k,j)=terrain(i,j)*g
        ph(i,k,j)=z*g-phb(i,k,j)
      enddo
    enddo
  enddo
  bounds=[1-ml,nx-ml,1-mb,ny-mb]
  do m=1,12
    rx=1+modulo(m,4);ry=1+modulo(m+1,3)
    nfx=(nx-1)*rx;nfy=(ny-1)*ry
    fl=1-fh;fr=nfx+fh;fb=1-fh;ft=nfy+fh
    allocate(uf(fl:fr,fb:ft),vf(fl:fr,fb:ft),frough(fl:fr,fb:ft))
    do j=fb,ft
      do i=fl,fr
        frough(i,j)=0.023+0.011*modulo(i*3+j*7,21)
      enddo
    enddo
    height=6.5
    select case(modulo(m-1,6))
      case(1);height=0.035
      case(2);height=38.
      case(3);height=260.
      case(4);height=4.
      case(5);height=110.
    end select
    fire_lsm_zcoupling=m>6
    ua=-777.;va=-777.;uf=-777.;vf=-777.
    call interpolate_atm2fire(0,height,1,nx,1,nz+1,1,ny, &
      ml,mh,1,nz+1,mb,mt,1,nx,1,ny,1,nx,1,ny, &
      1,nfx,1,nfy,fl,fr,fb,ft,1,nfx,1,nfy,1,nfx,1,nfy,rx,ry, &
      0.15,-0.12,u,v,ph,phb,rough,terrain,ua,va,uf,vf,frough)
    fbounds=[1-fl,nfx-fl,1-fb,nfy-fb]
    write(name,'(a,i0)') 'wind/native_',m
    call oracle_open(trim(name))
    call save_input()
    call oracle_put('uf',uf);call oracle_put('vf',vf)
    call oracle_put('uah',ua(1:nx+1,1:ny+1));call oracle_put('vah',va(1:nx+1,1:ny+1))
    call oracle_close()
    ! Four rectangular tiles exercise coarse halos and refinement offsets.
    atiles(:,1)=[1,4,1,3];atiles(:,2)=[5,nx,1,3]
    atiles(:,3)=[1,4,4,ny];atiles(:,4)=[5,nx,4,ny]
    ftiles(:,1)=[1,4*rx,1,3*ry];ftiles(:,2)=[4*rx+1,nfx,1,3*ry]
    ftiles(:,3)=[1,4*rx,3*ry+1,nfy];ftiles(:,4)=[4*rx+1,nfx,3*ry+1,nfy]
    ua=-777.;va=-777.;uf=-777.;vf=-777.
    do k=1,4
      tl=atiles(1,k);th=atiles(2,k);tb=atiles(3,k);tt=atiles(4,k)
      lft=ftiles(1,k);rft=ftiles(2,k);bft=ftiles(3,k);tft=ftiles(4,k)
      call interpolate_atm2fire(0,height,1,nx,1,nz+1,1,ny, &
        ml,mh,1,nz+1,mb,mt,1,nx,1,ny,tl,th,tb,tt, &
        1,nfx,1,nfy,fl,fr,fb,ft,1,nfx,1,nfy,lft,rft,bft,tft,rx,ry, &
        0.15,-0.12,u,v,ph,phb,rough,terrain,ua,va,uf,vf,frough)
    enddo
    write(name,'(a,i0)') 'wind/tiled_',m
    call oracle_open(trim(name))
    call save_input()
    do k=1,4
      atiles(:,k)=atiles(:,k)-[ml,ml,mb,mb]
      ftiles(:,k)=ftiles(:,k)-[fl,fl,fb,fb]
    enddo
    call oracle_put('tiles',atiles);call oracle_put('fine_tiles',ftiles)
    call oracle_put('uf',uf);call oracle_put('vf',vf)
    call oracle_put('uah',ua(1:nx+1,1:ny+1));call oracle_put('vah',va(1:nx+1,1:ny+1))
    call oracle_close()
    deallocate(uf,vf,frough)
  enddo
  ! Compact atmospheric allocations supply mass cells and terminal wind faces.
  ! Pad their horizontal scalar fields exactly as the device adapter does.
  do j=mb,mt
    do i=ml,mh
      do k=1,nz+1
        u(i,k,j)=u(max(1,min(nx,i)),k,max(1,min(ny-1,j)))
        v(i,k,j)=v(max(1,min(nx-1,i)),k,max(1,min(ny,j)))
        ph(i,k,j)=ph(max(1,min(nx-1,i)),k,max(1,min(ny-1,j)))
        phb(i,k,j)=phb(max(1,min(nx-1,i)),k,max(1,min(ny-1,j)))
      enddo
      rough(i,j)=rough(max(1,min(nx-1,i)),max(1,min(ny-1,j)))
      terrain(i,j)=terrain(max(1,min(nx-1,i)),max(1,min(ny-1,j)))
    enddo
  enddo
  do m=1,9
    if(m==7) then
      do j=mb,mt
        do i=ml,mh
          phb(i,:,j)=phb(1,:,1)
        enddo
      enddo
    endif
    rx=1+modulo(m,4);ry=1+modulo(m+1,3)
    nfx=(nx-1)*rx;nfy=(ny-1)*ry
    fl=1-fh;fr=nfx+fh;fb=1-fh;ft=nfy+fh
    allocate(uf(fl:fr,fb:ft),vf(fl:fr,fb:ft),frough(fl:fr,fb:ft))
    do j=fb,ft
      do i=fl,fr
        frough(i,j)=0.023+0.011*modulo(i*3+j*7,21)
      enddo
    enddo
    height=6.5+19.*(m-1)
    fire_lsm_zcoupling=m>3
    if(m==9) then
      u=0.;v=0.
      u(nx,:,:)= -0.
      v(:,:,ny)= -0.
      height=10000.
      fire_lsm_zcoupling=.false.
    endif
    ua=-777.;va=-777.;uf=-777.;vf=-777.
    call interpolate_atm2fire(0,height,1,nx,1,nz+1,1,ny, &
      ml,mh,1,nz+1,mb,mt,1,nx,1,ny,1,nx,1,ny, &
      1,nfx,1,nfy,fl,fr,fb,ft,1,nfx,1,nfy,1,nfx,1,nfy,rx,ry, &
      0.15,-0.12,u,v,ph,phb,rough,terrain,ua,va,uf,vf,frough)
    fbounds=[1-fl,nfx-fl,1-fb,nfy-fb]
    write(name,'(a,i0)') 'wind/compact_',m
    call oracle_open(trim(name))
    call save_input()
    call oracle_put('uf',uf);call oracle_put('vf',vf)
    call oracle_put('uah',ua(1:nx-1,1:ny-1));call oracle_put('vah',va(1:nx-1,1:ny-1))
    call oracle_put('uah_staggered',ua(1:nx,1:ny-1));call oracle_put('vah_staggered',va(1:nx-1,1:ny))
    call oracle_close()
    ! The coupled driver passes inclusive mass-cell bounds, excluding its
    ! terminal faces, while retaining all refined physical fire cells.
    ua=-777.;va=-777.;uf=-777.;vf=-777.
    call interpolate_atm2fire(0,height,1,nx-1,1,nz+1,1,ny-1, &
      ml,mh,1,nz+1,mb,mt,1,nx-1,1,ny-1,1,nx-1,1,ny-1, &
      1,nfx,1,nfy,fl,fr,fb,ft,1,nfx,1,nfy,1,nfx,1,nfy,rx,ry, &
      0.15,-0.12,u,v,ph,phb,rough,terrain,ua,va,uf,vf,frough)
    bounds=[1-ml,nx-1-ml,1-mb,ny-1-mb]
    write(name,'(a,i0)') 'wind/drivercompact_',m
    call oracle_open(trim(name))
    call save_input()
    call oracle_put('uf',uf);call oracle_put('vf',vf)
    call oracle_put('uah',ua(1:nx-1,1:ny-1));call oracle_put('vah',va(1:nx-1,1:ny-1))
    call oracle_put('uah_staggered',ua(1:nx,1:ny-1));call oracle_put('vah_staggered',va(1:nx-1,1:ny))
    call oracle_close()
    bounds=[1-ml,nx-ml,1-mb,ny-mb]
    deallocate(uf,vf,frough)
  enddo
contains
  subroutine save_input()
    call oracle_put('u',u);call oracle_put('v',v)
    call oracle_put('ph',ph);call oracle_put('phb',phb)
    call oracle_put('z0',rough);call oracle_put('zs',terrain);call oracle_put('z0f',frough)
    call oracle_put('domain',bounds);call oracle_put('fine_domain',fbounds)
    call oracle_put('sr_x',rx);call oracle_put('sr_y',ry)
    call oracle_put('fire_wind_height',height)
    call oracle_put('fire_lsm_zcoupling',merge(1,0,fire_lsm_zcoupling))
    call oracle_put('fire_lsm_zcoupling_ref',fire_lsm_zcoupling_ref)
  end subroutine
end program
