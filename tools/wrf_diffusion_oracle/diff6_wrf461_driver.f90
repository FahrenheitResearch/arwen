! Stream driver for the byte-unmodified WRF v4.6.1 sixth_order_diffusion routine.
!
! WRF's own scalar chain is formed here in REAL, not handed in pre-rounded:
!   dt     = real(time_step) + real(num)/real(den)   (share/set_timekeeping.F:343)
!   dt_rk  = grid%dt/3.                              (dyn_em/solve_em.F:599, rk_step 1)
!   rdx    = 1./config_flags%dx                      (dyn_em/module_initialize_real.F:3746)
! The u/v/w/theta rows receive dt (rk_tendency, module_em.F:882-919); the
! moist/scalar/tke rows receive dt_rk as dt_step (rk_scalar_tend, :1421-1425).
! All six directional map factors are passed separately, as solve_em does.
! Files are opened little-endian so WRF's -fconvert=big-endian build flag
! cannot reinterpret the stream.
program diff6_wrf461_driver
  use module_big_step_utilities_em, only: sixth_order_diffusion
  use module_configure, only: grid_config_rec_type
  implicit none
  type(grid_config_rec_type) :: config
  integer :: nx,ny,nz,opt,slope,oxs,oxe,oys,oye,spec,nest,ts,num,den,scalar_row,io
  character(1024) :: infile,outfile
  character(1) :: name
  real :: dt,dt_arg,factor,dx,dy,rdx,rdy,thresh
  real,allocatable :: field(:,:,:),tend(:,:,:),mut(:,:),c1(:),c2(:)
  real,allocatable :: phb(:,:,:),ph(:,:,:),mtx(:,:),mty(:,:),mux(:,:),muy(:,:),mvx(:,:),mvy(:,:)
  call get_command_argument(1,infile)
  call get_command_argument(2,outfile)
  open(newunit=io,file=trim(infile),access='stream',form='unformatted',status='old',convert='little_endian')
  read(io) nx,ny,nz,opt,slope,oxs,oxe,oys,oye,spec,nest,ts,num,den,scalar_row
  read(io) name
  read(io) factor,dx,dy,thresh
  allocate(field(-3:nx+3,1:nz+1,-3:ny+3),tend(-3:nx+3,1:nz+1,-3:ny+3))
  allocate(phb(-3:nx+3,1:nz+1,-3:ny+3),ph(-3:nx+3,1:nz+1,-3:ny+3))
  allocate(mut(-3:nx+3,-3:ny+3),c1(1:nz+1),c2(1:nz+1))
  allocate(mtx(-3:nx+3,-3:ny+3),mty(-3:nx+3,-3:ny+3),mux(-3:nx+3,-3:ny+3), &
           muy(-3:nx+3,-3:ny+3),mvx(-3:nx+3,-3:ny+3),mvy(-3:nx+3,-3:ny+3))
  read(io) field,tend,mut,c1,c2,phb,mtx,mty,mux,muy,mvx,mvy
  close(io)
  ph=0.
  dt = real(ts) + real(num)/real(den)
  dt_arg = dt
  if (scalar_row == 1) dt_arg = dt/3.
  rdx = 1./dx
  rdy = 1./dy
  config%specified=spec==1
  config%nested=nest==1
  config%open_xs=oxs==1
  config%open_xe=oxe==1
  config%open_ys=oys==1
  config%open_ye=oye==1
  config%diff_6th_slopeopt=slope
  config%diff_6th_thresh=thresh
  call sixth_order_diffusion(name,field,tend,mut,dt_arg,config,c1,c2,opt,factor,phb,ph,rdx,rdy, &
                            mtx,mty,mux,muy,mvx,mvy, &
                            0,nx,0,ny,1,nz+1,-3,nx+3,-3,ny+3,1,nz+1,0,nx,0,ny,1,nz+1)
  open(newunit=io,file=trim(outfile),access='stream',form='unformatted',status='replace',convert='little_endian')
  write(io) dt_arg,rdx,rdy
  write(io) tend
  close(io)
end program
