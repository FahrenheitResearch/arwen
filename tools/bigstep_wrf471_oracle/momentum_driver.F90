program momentum_driver
use module_big_step_utilities_em
implicit none
integer :: nx,ny,nz,mode,bx,by,lid,n3,n2,n1,ios
integer :: ims,ime,jms,jme,kms,kme
character(len=1024) :: input_file,output_file
type(grid_config_rec_type) :: config
real,allocatable :: a(:,:,:,:),b(:,:,:),z(:,:)
real :: scalars(7)
call get_command_argument(1,input_file)
call get_command_argument(2,output_file)
open(10,file=trim(input_file),access='stream',form='unformatted',status='old')
read(10) nx,ny,nz,mode,bx,by,lid
ims=-1;ime=nx+2;jms=-1;jme=ny+2;kms=1;kme=nz+1
allocate(a(ims:ime,kms:kme,jms:jme,19),b(ims:ime,jms:jme,12),z(kms:kme,7))
read(10) a,b,z,scalars
close(10)
config%specified=.false.;config%nested=.false.;config%polar=.false.;config%map_proj=1
config%open_xs=bx/=0;config%open_xe=bx/=0
config%open_ys=by/=0;config%open_ye=by/=0
config%periodic_x=bx==0
select case(mode)
case(1,5)
call horizontal_pressure_gradient(a(:,:,:,17),a(:,:,:,18), &
 a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),a(:,:,:,5),a(:,:,:,6),a(:,:,:,7),a(:,:,:,8), &
 b(:,:,1),b(:,:,2),b(:,:,3),z(:,1),z(:,2),z(:,3),z(:,4),z(:,5), &
 scalars(1),scalars(2),scalars(3),scalars(4),scalars(5),scalars(6),scalars(7), &
 b(:,:,4),b(:,:,4),b(:,:,5),b(:,:,5),b(:,:,6),b(:,:,6),config,.true.,lid/=0, &
 1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
case(2,4)
call coriolis(a(:,:,:,9),a(:,:,:,10),a(:,:,:,11),a(:,:,:,17),a(:,:,:,18),a(:,:,:,19),config, &
 b(:,:,6),b(:,:,6),b(:,:,4),b(:,:,4),b(:,:,5),b(:,:,5), &
 b(:,:,7),b(:,:,8),b(:,:,9),b(:,:,10),z(:,6),z(:,7), &
 1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
end select
if(mode==3.or.mode==4) then
call curvature(a(:,:,:,9),a(:,:,:,10),a(:,:,:,11),a(:,:,:,12),a(:,:,:,13),a(:,:,:,14), &
 a(:,:,:,17),a(:,:,:,18),a(:,:,:,19),config, &
 b(:,:,4),b(:,:,4),b(:,:,5),b(:,:,5),b(:,:,6),b(:,:,6),b(:,:,11), &
 z(:,6),z(:,7),scalars(6),scalars(7), &
 1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
endif
open(11,file=trim(output_file),access='stream',form='unformatted',status='replace')
write(11) a(:,:,:,17:19)
close(11)
end program
