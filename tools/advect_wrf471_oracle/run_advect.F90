! Binary caller for the byte-unmodified WRF v4.7.1 advection routines.
! Array declarations and complete argument lists retain WRF's (i,k,j) order.
program run_advect
  use iso_fortran_env, only: int32, real32, error_unit
  use module_advect_em, only: advect_u, advect_v, advect_w, advect_scalar, &
       advect_scalar_pd, advect_scalar_mono
  use module_configure, only: grid_config_rec_type
  implicit none
  integer(int32) :: hdr(27)
  integer :: mode, step, flags, input, output, position, input_size
  integer :: ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme
  integer :: its,ite,jts,jte,kts,kte
  logical :: tenddec
  real(real32) :: rdx,rdy,dt
  real(real32), allocatable :: field(:,:,:), old(:,:,:), tendency(:,:,:)
  real(real32), allocatable :: ht(:,:,:),zt(:,:,:),ru(:,:,:),rv(:,:,:),rom(:,:,:),romi(:,:,:)
  real(real32), allocatable :: mut(:,:),mub(:,:),mu_old(:,:)
  real(real32), allocatable :: msfux(:,:),msfuy(:,:),msfvx(:,:),msfvy(:,:),msftx(:,:),msfty(:,:)
  real(real32), allocatable :: c1(:),c2(:),fzm(:),fzp(:),rdzw(:),rdzu(:)
  character(2048) :: in_path,out_path
  type(grid_config_rec_type) :: cfg

  if (command_argument_count() /= 2) error stop 'usage: run_advect INPUT.bin OUTPUT.bin'
  if (storage_size(1.0) /= 32) error stop 'WRF default REAL must be binary32'
  if (storage_size(1) /= 32) error stop 'WRF default INTEGER must be int32'
  call get_command_argument(1,in_path)
  call get_command_argument(2,out_path)
  open(newunit=input,file=trim(in_path),form='unformatted',access='stream',status='old',action='read')
  read(input) hdr
  if (hdr(1) /= 1) error stop 'unknown advection protocol version'
  mode=hdr(2)
  ids=hdr(3);ide=hdr(4);jds=hdr(5);jde=hdr(6);kds=hdr(7);kde=hdr(8)
  ims=hdr(9);ime=hdr(10);jms=hdr(11);jme=hdr(12);kms=hdr(13);kme=hdr(14)
  its=hdr(15);ite=hdr(16);jts=hdr(17);jte=hdr(18);kts=hdr(19);kte=hdr(20)
  step=hdr(21);flags=hdr(26);tenddec=hdr(27)/=0
  cfg%h_mom_adv_order=hdr(22);cfg%v_mom_adv_order=hdr(23)
  cfg%h_sca_adv_order=hdr(24);cfg%v_sca_adv_order=hdr(25)
  cfg%periodic_x=btest(flags,0);cfg%periodic_y=btest(flags,1)
  cfg%specified=btest(flags,2);cfg%nested=btest(flags,3)
  cfg%open_xs=btest(flags,4);cfg%open_xe=btest(flags,5)
  cfg%open_ys=btest(flags,6);cfg%open_ye=btest(flags,7)
  cfg%symmetric_xs=btest(flags,8);cfg%symmetric_xe=btest(flags,9)
  cfg%symmetric_ys=btest(flags,10);cfg%symmetric_ye=btest(flags,11)
  cfg%polar=btest(flags,12)
  read(input) rdx,rdy,dt
  if (any([ime<ims,jme<jms,kme<kms,ite<its,jte<jts,kte<kts])) error stop 'invalid dimensions'
  allocate(field(ims:ime,kms:kme,jms:jme),old(ims:ime,kms:kme,jms:jme), &
       tendency(ims:ime,kms:kme,jms:jme),ht(ims:ime,kms:kme,jms:jme), &
       zt(ims:ime,kms:kme,jms:jme),ru(ims:ime,kms:kme,jms:jme), &
       rv(ims:ime,kms:kme,jms:jme),rom(ims:ime,kms:kme,jms:jme),romi(ims:ime,kms:kme,jms:jme))
  allocate(mut(ims:ime,jms:jme),mub(ims:ime,jms:jme),mu_old(ims:ime,jms:jme), &
       msfux(ims:ime,jms:jme),msfuy(ims:ime,jms:jme),msfvx(ims:ime,jms:jme), &
       msfvy(ims:ime,jms:jme),msftx(ims:ime,jms:jme),msfty(ims:ime,jms:jme))
  allocate(c1(kms:kme),c2(kms:kme),fzm(kms:kme),fzp(kms:kme),rdzw(kms:kme),rdzu(kms:kme))
  read(input) field,old,tendency,ru,rv,rom,romi
  read(input) mut,mub,mu_old,msfux,msfuy,msfvx,msfvy,msftx,msfty
  read(input) c1,c2,fzm,fzp,rdzw,rdzu
  inquire(unit=input,pos=position,size=input_size)
  if (position /= input_size+1) error stop 'extra trailing input data'
  close(input)
  ht=-999999.0_real32
  zt=-999999.0_real32

  select case(mode)
  case(1)
    call advect_scalar(field,old,tendency,ru,rv,rom,c1,c2,mut,step,cfg, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzw, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case(2)
    call advect_u(field,old,tendency,ru,rv,rom,c1,c2,mut,step,cfg, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzw, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case(3)
    call advect_v(field,old,tendency,ru,rv,rom,c1,c2,mut,step,cfg, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzw, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case(4)
    call advect_w(field,old,tendency,ru,rv,rom,c1,c2,mut,step,cfg, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzu, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case(5)
    call advect_scalar_pd(field,old,tendency,ht,zt,ru,rv,rom,c1,c2,mut,mub,mu_old,step,cfg,tenddec, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzw,dt, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case(6)
    call advect_scalar_mono(field,old,tendency,ht,zt,ru,rv,rom,romi,c1,c2,mut,mub,mu_old,cfg,tenddec, &
         msfux,msfuy,msfvx,msfvy,msftx,msfty,fzm,fzp,rdx,rdy,rdzw,dt, &
         ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  case default
    write(error_unit,*) 'unknown routine mode: ',mode
    error stop 2
  end select
  open(newunit=output,file=trim(out_path),form='unformatted',access='stream',status='replace',action='write')
  write(output) tendency,ht,zt
  close(output)
end program run_advect
