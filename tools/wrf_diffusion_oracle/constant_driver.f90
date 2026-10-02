program constant_driver
  use module_big_step_utilities_em, only: horizontal_diffusion, horizontal_diffusion_3dmp, &
        vertical_diffusion,vertical_diffusion_3dmp,vertical_diffusion_u,vertical_diffusion_v
  use module_configure, only: grid_config_rec_type
  implicit none
  type(grid_config_rec_type) :: cfg
  integer :: nx,ny,nz,op,mode,io
  character(1024)::input_file,output_file
  character(1)::name
  real :: kh,kv,rdx,rdy
  real,allocatable::f(:,:,:),t(:,:,:),base(:,:,:),mut(:,:),alt(:,:,:),km(:,:,:),maps(:,:)
  real,allocatable::c1(:),c2(:),rdn(:),rdnw(:),base1d(:)
  call get_command_argument(1,input_file);call get_command_argument(2,output_file)
  open(newunit=io,file=trim(input_file),access='stream',form='unformatted',status='old')
  read(io)nx,ny,nz,op,mode;read(io)name;read(io)kh,kv,rdx,rdy
  allocate(f(-3:nx+3,1:nz+1,-3:ny+3),t(-3:nx+3,1:nz+1,-3:ny+3),base(-3:nx+3,1:nz+1,-3:ny+3), &
           alt(-3:nx+3,1:nz+1,-3:ny+3),km(-3:nx+3,1:nz+1,-3:ny+3),mut(-3:nx+3,-3:ny+3),maps(-3:nx+3,-3:ny+3), &
           c1(nz+1),c2(nz+1),rdn(nz+1),rdnw(nz+1),base1d(nz+1))
  read(io)f,mut,alt,rdn,rdnw,c1,c2
  close(io)
  t=0.;base=0.;base1d=0.;km=kh;maps=1.
  cfg%specified=mode==1
  cfg%periodic_x=mode==0
  if(op==0)then
    if(name=='m')then
      call horizontal_diffusion_3dmp(name,f,t,mut,c1,c2,cfg,base,maps,maps,maps,maps,maps,maps,maps, &
           kh,km,rdx,rdy,0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    else
      call horizontal_diffusion(name,f,t,mut,c1,c2,cfg,maps,maps,maps,maps,maps,maps,maps, &
           kh,km,rdx,rdy,0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    endif
  else
    if(name=='u')then
      call vertical_diffusion_u(f,t,cfg,base1d,c1,c2,alt,mut,rdn,rdnw,kv, &
           0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    else if(name=='v')then
      call vertical_diffusion_v(f,t,cfg,base1d,c1,c2,alt,mut,rdn,rdnw,kv, &
           0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    else if(name=='m')then
      call vertical_diffusion_3dmp(f,t,cfg,base,c1,c2,alt,mut,rdn,rdnw,kv, &
           0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    else
      call vertical_diffusion(name,f,t,cfg,c1,c2,alt,mut,rdn,rdnw,kv, &
           0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
    endif
  endif
  open(newunit=io,file=trim(output_file),access='stream',form='unformatted',status='replace')
  write(io)t
  close(io)
end program
