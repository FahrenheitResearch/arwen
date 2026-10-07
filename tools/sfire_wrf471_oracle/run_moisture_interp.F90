program run_moisture_interp
  use module_fr_fire_phys
  use module_fr_fire_util
  use oracle_io
  implicit none
  integer,parameter :: ax=6,ay=5,fx=10,fy=8,nc=5
  real :: classes(0:ax+1,nc,0:ay+1),cat(0:fx+1,0:fy+1),out(0:fx+1,0:fy+1)
  real :: field(0:ax+1,0:ay+1),interpolated(0:fx+1,0:fy+1),all_classes(0:fx+1,nc,0:fy+1)
  integer :: i,j,k
  character(len=1024) :: root
  call oracle_root(root)
  fire_print_msg=0;fire_print_file=0
  call init_fuel_cats(.true.)
  do j=0,ay+1
    do k=1,nc
      do i=0,ax+1
        classes(i,k,j)=real(k)*.025+real(i*i+j*j)*.001+real(i*j)*.0003
      enddo
    enddo
  enddo
  do j=0,fy+1
    do i=0,fx+1
      cat(i,j)=real(mod(i+j,14)+1)
    enddo
  enddo
  all_classes=-999.;out=-999.
  do k=1,nc
    field=classes(:,k,:);interpolated=-999.
    call interpolate_z2fire(0,1,ax,1,ay,0,ax+1,0,ay+1,1,ax,1,ay,1,ax,1,ay, &
                           1,fx,1,fy,0,fx+1,0,fy+1,1,fx,1,fy,2,2,field,interpolated,0)
    all_classes(:,k,:)=interpolated
  enddo
  call fuel_moisture(0,nc,1,ax,1,ay,0,ax+1,0,ay+1,1,ax,1,ay,1,ax,1,ay, &
                    1,fx,1,fy,0,fx+1,0,fy+1,1,fx,1,fy,2,2,cat,classes,out)
  call oracle_open('moisture_interp')
  call oracle_put('classes_atm_in',classes);call oracle_put('nfuel_cat_in',cat)
  call oracle_put('classes_fire_out',all_classes);call oracle_put('fmc_g_out',out)
  call oracle_close()
end program run_moisture_interp
