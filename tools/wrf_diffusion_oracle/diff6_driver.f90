! Stream driver for the unmodified WRF sixth_order_diffusion routine.
program diff6_driver
  use module_big_step_utilities_em, only: sixth_order_diffusion
  use module_configure, only: grid_config_rec_type
  implicit none
  type(grid_config_rec_type) :: config
  integer :: nx,ny,nz,opt,slope,mode,io
  character(1024) :: infile,outfile
  character(1) :: name
  real :: dt,factor,rdx,rdy,thresh
  real,allocatable :: field(:,:,:),tend(:,:,:),mut(:,:),c1(:),c2(:)
  real,allocatable :: phb(:,:,:),ph(:,:,:),mtx(:,:),mty(:,:),mux(:,:),muy(:,:),mvx(:,:),mvy(:,:)
  call get_command_argument(1,infile)
  call get_command_argument(2,outfile)
  open(newunit=io,file=trim(infile),access='stream',form='unformatted',status='old')
  read(io) nx,ny,nz,opt,slope,mode
  read(io) name
  read(io) dt,factor,rdx,rdy,thresh
  allocate(field(-3:nx+3,1:nz+1,-3:ny+3),tend(-3:nx+3,1:nz+1,-3:ny+3))
  allocate(phb(-3:nx+3,1:nz+1,-3:ny+3),ph(-3:nx+3,1:nz+1,-3:ny+3))
  allocate(mut(-3:nx+3,-3:ny+3),c1(1:nz+1),c2(1:nz+1))
  allocate(mtx(-3:nx+3,-3:ny+3),mty(-3:nx+3,-3:ny+3),mux(-3:nx+3,-3:ny+3), &
           muy(-3:nx+3,-3:ny+3),mvx(-3:nx+3,-3:ny+3),mvy(-3:nx+3,-3:ny+3))
  read(io) field,tend,mut,c1,c2,phb,mtx,mty,mux,muy,mvx,mvy
  close(io)
  ph=0.
  config%specified=mode==1
  config%nested=mode==2
  config%open_xs=mode==3
  config%open_xe=mode==3
  config%open_ys=mode==3
  config%open_ye=mode==3
  config%diff_6th_slopeopt=slope
  config%diff_6th_thresh=thresh
  call sixth_order_diffusion(name,field,tend,mut,dt,config,c1,c2,opt,factor,phb,ph,rdx,rdy, &
                            mtx,mty,mux,muy,mvx,mvy, &
                            0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
  open(newunit=io,file=trim(outfile),access='stream',form='unformatted',status='replace')
  write(io) tend
  close(io)
end program
