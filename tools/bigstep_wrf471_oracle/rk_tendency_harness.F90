! Call native WRF RK preparation and the unchanged RK tendency routine.
program rk_tendency_harness
  use rk_full_reference, only: rk_step_prep,rk_tendency
  use module_configure, only: grid_config_rec_type
  use module_state_description, only: p_qv,p_qc,p_qr,p_qi,p_qs,p_qg
  implicit none
  type(grid_config_rec_type) :: config
  integer :: nx,ny,nz,step,unit,ierr
  character(len=4096) :: input_file,output_file
  real,allocatable :: x(:,:,:,:),m(:,:,:),z(:,:),q(:,:,:,:)
  real,allocatable :: tend(:,:,:,:),held(:,:,:,:),saved(:,:,:,:),prep(:,:,:,:),before_prep(:,:,:,:)
  real,allocatable :: mass(:,:,:),misc(:,:,:,:),mtend(:,:,:)
  real :: coeff(5),cfl(2)
  call get_command_argument(1,input_file)
  call get_command_argument(2,output_file)
  open(newunit=unit,file=trim(input_file),access='stream',form='unformatted',status='old')
  read(unit) nx,ny,nz,step
  allocate(x(-3:nx+4,nz+1,-3:ny+4,11),m(-3:nx+4,-3:ny+4,15),z(nz+1,14))
  allocate(q(-3:nx+4,nz+1,-3:ny+4,7))
  read(unit) x,m,z,q,coeff
  close(unit)
  allocate(tend(-3:nx+4,nz+1,-3:ny+4,5),held(-3:nx+4,nz+1,-3:ny+4,5))
  allocate(saved(-3:nx+4,nz+1,-3:ny+4,5),prep(-3:nx+4,nz+1,-3:ny+4,9))
  allocate(before_prep(-3:nx+4,nz+1,-3:ny+4,9))
  allocate(mass(-3:nx+4,-3:ny+4,3),misc(-3:nx+4,nz+1,-3:ny+4,5))
  allocate(mtend(-3:nx+4,-3:ny+4,2))
  tend=0.;held=0.;saved=0.;prep=0.;mass=0.;misc=0.;mtend=0.
  ! The generated record contains intrinsic fixed-size fields only.
  ! Zero initialization disables optional branches, then every active setting
  ! is specified explicitly below.
  config=transfer(repeat(achar(0),storage_size(config)/8),config)
  config%specified=.true.;config%periodic_x=.false.;config%periodic_y=.false.
  config%rk_ord=3;config%h_mom_adv_order=5;config%h_sca_adv_order=5
  config%v_mom_adv_order=3;config%v_sca_adv_order=3
  config%diff_opt=2;config%km_opt=4;config%map_proj=1
  config%phi_adv_z=1;config%w_crit_cfl=1.
  p_qv=2;p_qc=3;p_qr=4;p_qi=5;p_qs=6;p_qg=7
  call nl_set_time_step(1,20)
  call rk_step_prep(config,step,x(:,:,:,1),x(:,:,:,2),x(:,:,:,3),x(:,:,:,4),x(:,:,:,5),m(:,:,1), &
       z(:,1),z(:,2),z(:,3),z(:,4),q, &
       prep(:,:,:,1),prep(:,:,:,2),prep(:,:,:,3),prep(:,:,:,4),prep(:,:,:,5),prep(:,:,:,6), &
       mass(:,:,1),mass(:,:,2),m(:,:,2),mass(:,:,3),x(:,:,:,6),x(:,:,:,7),x(:,:,:,8), &
       x(:,:,:,9),x(:,:,:,10),prep(:,:,:,7),prep(:,:,:,8),prep(:,:,:,9), &
       m(:,:,5),m(:,:,6),m(:,:,7),m(:,:,8),m(:,:,9),m(:,:,3),m(:,:,4), &
       z(:,5),z(:,6),z(:,7),1./3000.,1./3000.,7, &
       1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  before_prep=prep
  call rk_tendency(config,step,tend(:,:,:,1),tend(:,:,:,2),tend(:,:,:,3),tend(:,:,:,4),tend(:,:,:,5), &
       held(:,:,:,1),held(:,:,:,2),held(:,:,:,3),held(:,:,:,4),held(:,:,:,5), &
       mtend(:,:,1),saved(:,:,:,1),saved(:,:,:,2),saved(:,:,:,3),saved(:,:,:,4),saved(:,:,:,5), &
       mtend(:,:,2),misc(:,:,:,1),prep(:,:,:,1),prep(:,:,:,2),prep(:,:,:,3),prep(:,:,:,4), &
       misc(:,:,:,2),misc(:,:,:,3),x(:,:,:,1),x(:,:,:,2),x(:,:,:,3),x(:,:,:,4),x(:,:,:,5), &
       x(:,:,:,1),x(:,:,:,2),x(:,:,:,3),x(:,:,:,4),x(:,:,:,5),misc(:,:,:,4),x(:,:,:,6),x(:,:,:,11), &
       m(:,:,1),m(:,:,1),mass(:,:,3),mass(:,:,1),mass(:,:,2),m(:,:,2), &
       z(:,1),z(:,2),z(:,3),z(:,4),x(:,:,:,9),m(:,:,15),prep(:,:,:,6), &
       x(:,:,:,8),x(:,:,:,7),prep(:,:,:,5),prep(:,:,:,7),prep(:,:,:,8),prep(:,:,:,9), &
       z(:,10),z(:,11),z(:,12),z(:,13),z(:,14), &
       m(:,:,5),m(:,:,6),m(:,:,7),m(:,:,8),m(:,:,9),m(:,:,3),m(:,:,4), &
       m(:,:,10),m(:,:,11),m(:,:,12),m(:,:,13),m(:,:,14), &
       z(:,5),z(:,6),z(:,8),z(:,9),20.,1./3000.,1./3000.,0.,0.,misc(:,:,:,5),misc(:,:,:,5), &
       0,0.,0,0.,0.,0,0,coeff(1),coeff(2),coeff(3),coeff(4),coeff(5),7,.true.,.true.,0.,0., &
       1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1,cfl(1),cfl(2))
  open(newunit=unit,file=trim(output_file),access='stream',form='unformatted',status='replace')
  write(unit) tend,held,saved,mtend,misc,prep,mass,cfl,before_prep
  close(unit)
end program
